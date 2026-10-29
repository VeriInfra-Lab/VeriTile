import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_seqused_dx
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
  let vindex_put_ptr : RegionName := "vindex_put"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vbmm_5_ptr : RegionName := "vbmm_5"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vneg_ptr : RegionName := "vneg"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vadd_8_ptr : RegionName := "vadd_8"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vcat_9_ptr : RegionName := "vcat_9"
  let vcat_10_ptr : RegionName := "vcat_10"
  let vmm_6_ptr : RegionName := "vmm_6"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vpow_5_ptr : RegionName := "vpow_5"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vcat_11_ptr : RegionName := "vcat_11"
  let vmm_7_ptr : RegionName := "vmm_7"
  let vmm_8_ptr : RegionName := "vmm_8"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vpow_6_ptr : RegionName := "vpow_6"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vpow_7_ptr : RegionName := "vpow_7"
  let vmul_43_ptr : RegionName := "vmul_43"
  let vmul_44_ptr : RegionName := "vmul_44"
  let vadd_14_ptr : RegionName := "vadd_14"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vmm_9_ptr : RegionName := "vmm_9"
  let vadd_15_ptr : RegionName := "vadd_15"
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
      r83_left = tl.load($(vto_copy_23_ptr) + (((((((r83_group % $(8)) * $(82944)) + (r83_lane[None, :] * $(288)) + r83_row[:, None]) / $(82944)) % $(8)) * $(82944)) + ((((((r83_group % $(8)) * $(82944)) + (r83_lane[None, :] * $(288)) + r83_row[:, None]) / $(288)) % $(288)) * $(288)) + ((((r83_group % $(8)) * $(82944)) + (r83_lane[None, :] * $(288)) + r83_row[:, None]) % $(288))), dtype=tl.bfloat16)
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
    tl.store($(vto_copy_24_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(663552))
    r86_lane = tl.arange($(288))
    r86_row_values = tl.load($(vsoftmax_ptr) + ((((r86_z[:, None] / $(82944)) % $(8)) * $(82944)) + (((r86_z[:, None] / $(288)) % $(288)) * $(288)) + (r86_lane[None, :] % $(288))), dtype=tl.float32)
    r86_row_grad = tl.load($(vto_copy_24_ptr) + ((((r86_z[:, None] / $(82944)) % $(8)) * $(82944)) + (((r86_z[:, None] / $(288)) % $(288)) * $(288)) + (r86_lane[None, :] % $(288))), dtype=tl.float32)
    r86_weighted = r86_row_grad * r86_row_values
    r86_dot_grad = tl.sum(r86_weighted, axis=1)
    r86_value = (tl.load($(vto_copy_24_ptr) + ((((r86_z / $(82944)) % $(8)) * $(82944)) + (((r86_z / $(288)) % $(288)) * $(288)) + (r86_z % $(288))), dtype=tl.float32) - r86_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r86_z / $(82944)) % $(8)) * $(82944)) + (((r86_z / $(288)) % $(288)) * $(288)) + (r86_z % $(288))), dtype=tl.float32)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(663552)])
    tl.store($(vsoftmax_backward_data_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(663552))
    r87_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r87_z / $(82944)) % $(8)) * $(82944)) + (((r87_z / $(288)) % $(288)) * $(288)) + (r87_z % $(288))), dtype=tl.float32)
    r87_value = r87_value.to(tl.bfloat16)
    r87_value = tl.broadcast_to(r87_value, [$(663552)])
    tl.store($(vto_copy_25_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(663552))
    r88_value = (tl.load($(vto_copy_25_ptr) + ((((r88_z / $(82944)) % $(8)) * $(82944)) + (((r88_z / $(288)) % $(288)) * $(288)) + (r88_z % $(288))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
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
    r90_row = tl.arange($(288))
    r90_col = tl.arange($(192))
    r90_lane = tl.arange($(288))
    for r90_group in range($(8)) {
      r90_left = tl.load($(vmul_22_ptr) + (((((((r90_group % $(8)) * $(82944)) + (r90_row[:, None] * $(288)) + r90_lane[None, :]) / $(82944)) % $(8)) * $(82944)) + ((((((r90_group % $(8)) * $(82944)) + (r90_row[:, None] * $(288)) + r90_lane[None, :]) / $(288)) % $(288)) * $(288)) + ((((r90_group % $(8)) * $(82944)) + (r90_row[:, None] * $(288)) + r90_lane[None, :]) % $(288))), dtype=tl.bfloat16)
      r90_right = tl.load($(vcat_8_ptr) + (((((((r90_group % $(8)) * $(55296)) + (r90_col[None, :] * $(288)) + r90_lane[:, None]) / $(55296)) % $(8)) * $(55296)) + (((((r90_group % $(8)) * $(55296)) + (r90_col[None, :] * $(288)) + r90_lane[:, None]) % $(288)) * $(192)) + (((((r90_group % $(8)) * $(55296)) + (r90_col[None, :] * $(288)) + r90_lane[:, None]) / $(288)) % $(192))), dtype=tl.bfloat16)
      r90_value = tl.dot(r90_left, r90_right)
      r90_value = r90_value.to(tl.bfloat16)
      tl.store($(vbmm_5_ptr) + ((r90_group * $(55296)) + (r90_row[:, None] * $(192)) + r90_col[None, :]), r90_value)
    }
    r91_z = tl.arange($(18432))
    r91_lane = tl.arange($(8))
    r91_value = tl.sum(tl.load($(vbmm_4_ptr) + ((r91_lane[None, :] * $(55296)) + ((r91_z[:, None] % $(64)) * $(288)) + ((r91_z[:, None] / $(64)) % $(288)) + $(36864)), dtype=tl.bfloat16), axis=1)
    r91_value = r91_value.to(tl.bfloat16)
    r91_value = tl.broadcast_to(r91_value, [$(18432)])
    tl.store($(vsum_1_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(9216))
    r92_value = (tl.load($(vsum_1_ptr) + ((((r92_z / $(32)) % $(288)) * $(64)) + (r92_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r92_z / $(32)) % $(288)) * $(64)) + (r92_z % $(32))), dtype=tl.bfloat16))
    r92_value = r92_value.to(tl.bfloat16)
    r92_value = tl.broadcast_to(r92_value, [$(9216)])
    tl.store($(vmul_23_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(9216))
    r93_value = (tl.load($(vsum_1_ptr) + ((((r93_z / $(32)) % $(288)) * $(64)) + (r93_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r93_z / $(32)) % $(288)) * $(64)) + (r93_z % $(32))), dtype=tl.bfloat16))
    r93_value = r93_value.to(tl.bfloat16)
    r93_value = tl.broadcast_to(r93_value, [$(9216)])
    tl.store($(vmul_24_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(9216))
    r94_value = -tl.load($(vsum_1_ptr) + ((((r94_z / $(32)) % $(288)) * $(64)) + (r94_z % $(32))), dtype=tl.bfloat16)
    r94_value = r94_value.to(tl.bfloat16)
    r94_value = tl.broadcast_to(r94_value, [$(9216)])
    tl.store($(vneg_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(9216))
    r95_value = (tl.load($(vneg_ptr) + ((((r95_z / $(32)) % $(288)) * $(32)) + (r95_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r95_z / $(32)) % $(288)) * $(64)) + (r95_z % $(32))), dtype=tl.bfloat16))
    r95_value = r95_value.to(tl.bfloat16)
    r95_value = tl.broadcast_to(r95_value, [$(9216)])
    tl.store($(vmul_25_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(9216))
    r96_value = (tl.load($(vmul_24_ptr) + ((((r96_z / $(32)) % $(288)) * $(32)) + (r96_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_25_ptr) + ((((r96_z / $(32)) % $(288)) * $(32)) + (r96_z % $(32))), dtype=tl.bfloat16))
    r96_value = r96_value.to(tl.bfloat16)
    r96_value = tl.broadcast_to(r96_value, [$(9216)])
    tl.store($(vadd_7_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(9216))
    r97_value = (tl.load($(vsum_1_ptr) + ((((r97_z / $(32)) % $(288)) * $(64)) + (r97_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r97_z / $(32)) % $(288)) * $(64)) + (r97_z % $(32))), dtype=tl.bfloat16))
    r97_value = r97_value.to(tl.bfloat16)
    r97_value = tl.broadcast_to(r97_value, [$(9216)])
    tl.store($(vmul_26_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(9216))
    r98_value = (tl.load($(vmul_23_ptr) + ((((r98_z / $(32)) % $(288)) * $(32)) + (r98_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_26_ptr) + ((((r98_z / $(32)) % $(288)) * $(32)) + (r98_z % $(32))), dtype=tl.bfloat16))
    r98_value = r98_value.to(tl.bfloat16)
    r98_value = tl.broadcast_to(r98_value, [$(9216)])
    tl.store($(vadd_8_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(73728))
    r99_value = (tl.load($(vbmm_5_ptr) + ((((r99_z / $(9216)) % $(8)) * $(55296)) + (((r99_z / $(32)) % $(288)) * $(192)) + (r99_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r99_z / $(32)) % $(288)) * $(64)) + (r99_z % $(32))), dtype=tl.bfloat16))
    r99_value = r99_value.to(tl.bfloat16)
    r99_value = tl.broadcast_to(r99_value, [$(73728)])
    tl.store($(vmul_27_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(73728))
    r100_value = (tl.load($(vbmm_5_ptr) + ((((r100_z / $(9216)) % $(8)) * $(55296)) + (((r100_z / $(32)) % $(288)) * $(192)) + (r100_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r100_z / $(32)) % $(288)) * $(64)) + (r100_z % $(32))), dtype=tl.bfloat16))
    r100_value = r100_value.to(tl.bfloat16)
    r100_value = tl.broadcast_to(r100_value, [$(73728)])
    tl.store($(vmul_28_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(73728))
    r101_value = -tl.load($(vbmm_5_ptr) + ((((r101_z / $(9216)) % $(8)) * $(55296)) + (((r101_z / $(32)) % $(288)) * $(192)) + (r101_z % $(32)) + $(128)), dtype=tl.bfloat16)
    r101_value = r101_value.to(tl.bfloat16)
    r101_value = tl.broadcast_to(r101_value, [$(73728)])
    tl.store($(vneg_1_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(73728))
    r102_value = (tl.load($(vneg_1_ptr) + ((((r102_z / $(9216)) % $(8)) * $(9216)) + (((r102_z / $(32)) % $(288)) * $(32)) + (r102_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r102_z / $(32)) % $(288)) * $(64)) + (r102_z % $(32))), dtype=tl.bfloat16))
    r102_value = r102_value.to(tl.bfloat16)
    r102_value = tl.broadcast_to(r102_value, [$(73728)])
    tl.store($(vmul_29_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(73728))
    r103_value = (tl.load($(vmul_28_ptr) + ((((r103_z / $(9216)) % $(8)) * $(9216)) + (((r103_z / $(32)) % $(288)) * $(32)) + (r103_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_29_ptr) + ((((r103_z / $(9216)) % $(8)) * $(9216)) + (((r103_z / $(32)) % $(288)) * $(32)) + (r103_z % $(32))), dtype=tl.bfloat16))
    r103_value = r103_value.to(tl.bfloat16)
    r103_value = tl.broadcast_to(r103_value, [$(73728)])
    tl.store($(vadd_9_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(73728))
    r104_value = (tl.load($(vbmm_5_ptr) + ((((r104_z / $(9216)) % $(8)) * $(55296)) + (((r104_z / $(32)) % $(288)) * $(192)) + (r104_z % $(32)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r104_z / $(32)) % $(288)) * $(64)) + (r104_z % $(32))), dtype=tl.bfloat16))
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(73728)])
    tl.store($(vmul_30_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(73728))
    r105_value = (tl.load($(vmul_27_ptr) + ((((r105_z / $(9216)) % $(8)) * $(9216)) + (((r105_z / $(32)) % $(288)) * $(32)) + (r105_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_30_ptr) + ((((r105_z / $(9216)) % $(8)) * $(9216)) + (((r105_z / $(32)) % $(288)) * $(32)) + (r105_z % $(32))), dtype=tl.bfloat16))
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(73728)])
    tl.store($(vadd_10_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(18432))
    r106_value = tl.where((((r106_z % $(64)) >= $(1)) & ((r106_z % $(64)) < $(64)) & (((r106_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_7_ptr) + ((((r106_z / $(64)) % $(288)) * $(32)) + ((((r106_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(18432)])
    tl.store($(vslice_backward_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(18432))
    r107_value = tl.where((((r107_z % $(64)) >= $(0)) & ((r107_z % $(64)) < $(64)) & (((r107_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_8_ptr) + ((((r107_z / $(64)) % $(288)) * $(32)) + ((((r107_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r107_value = r107_value.to(tl.bfloat16)
    r107_value = tl.broadcast_to(r107_value, [$(18432)])
    tl.store($(vslice_backward_1_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(18432))
    r108_value = (tl.load($(vslice_backward_ptr) + ((((r108_z / $(64)) % $(288)) * $(64)) + (r108_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_1_ptr) + ((((r108_z / $(64)) % $(288)) * $(64)) + (r108_z % $(64))), dtype=tl.bfloat16))
    r108_value = r108_value.to(tl.bfloat16)
    r108_value = tl.broadcast_to(r108_value, [$(18432)])
    tl.store($(vadd_11_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(147456))
    r109_value = tl.where((((r109_z % $(64)) >= $(1)) & ((r109_z % $(64)) < $(64)) & (((r109_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_9_ptr) + ((((r109_z / $(18432)) % $(8)) * $(9216)) + (((r109_z / $(64)) % $(288)) * $(32)) + ((((r109_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r109_value = r109_value.to(tl.bfloat16)
    r109_value = tl.broadcast_to(r109_value, [$(147456)])
    tl.store($(vslice_backward_2_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(147456))
    r110_value = tl.where((((r110_z % $(64)) >= $(0)) & ((r110_z % $(64)) < $(64)) & (((r110_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_10_ptr) + ((((r110_z / $(18432)) % $(8)) * $(9216)) + (((r110_z / $(64)) % $(288)) * $(32)) + ((((r110_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r110_value = r110_value.to(tl.bfloat16)
    r110_value = tl.broadcast_to(r110_value, [$(147456)])
    tl.store($(vslice_backward_3_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(147456))
    r111_value = (tl.load($(vslice_backward_2_ptr) + ((((r111_z / $(18432)) % $(8)) * $(18432)) + (((r111_z / $(64)) % $(288)) * $(64)) + (r111_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_3_ptr) + ((((r111_z / $(18432)) % $(8)) * $(18432)) + (((r111_z / $(64)) % $(288)) * $(64)) + (r111_z % $(64))), dtype=tl.bfloat16))
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(147456)])
    tl.store($(vadd_12_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(589824))
    r112_value = tl.where((r112_z % $(256)) < $(128), tl.load($(vbmm_4_ptr) + ((((((((r112_z / $(73728)) % $(8)) * $(55296)) + (((r112_z % $(256)) - $(0)) * $(288)) + ((r112_z / $(256)) % $(288))) / $(55296)) % $(8)) * $(55296)) + ((((((r112_z / $(73728)) % $(8)) * $(192)) + ((r112_z % $(256)) - $(0))) % $(192)) * $(288)) + ((r112_z / $(256)) % $(288))), dtype=tl.bfloat16), tl.load($(vbmm_2_ptr) + ((((((((r112_z / $(73728)) % $(8)) * $(36864)) + (((r112_z / $(256)) % $(288)) * $(128)) + ((r112_z % $(256)) - $(128))) / $(36864)) % $(8)) * $(36864)) + (((((((r112_z / $(73728)) % $(8)) * $(36864)) + (((r112_z / $(256)) % $(288)) * $(128)) + ((r112_z % $(256)) - $(128))) / $(128)) % $(288)) * $(128)) + (((((r112_z / $(73728)) % $(8)) * $(36864)) + (((r112_z / $(256)) % $(288)) * $(128)) + ((r112_z % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16))
    r112_value = r112_value.to(tl.bfloat16)
    r112_value = tl.broadcast_to(r112_value, [$(589824)])
    tl.store($(vcat_9_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(442368))
    r113_value = tl.where((r113_z % $(192)) < $(128), tl.load($(vbmm_5_ptr) + ((((((((r113_z / $(55296)) % $(8)) * $(55296)) + (((r113_z / $(192)) % $(288)) * $(192)) + ((r113_z % $(192)) - $(0))) / $(55296)) % $(8)) * $(55296)) + (((((((r113_z / $(55296)) % $(8)) * $(55296)) + (((r113_z / $(192)) % $(288)) * $(192)) + ((r113_z % $(192)) - $(0))) / $(192)) % $(288)) * $(192)) + (((((r113_z / $(55296)) % $(8)) * $(55296)) + (((r113_z / $(192)) % $(288)) * $(192)) + ((r113_z % $(192)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(vadd_12_ptr) + ((((r113_z / $(55296)) % $(8)) * $(18432)) + (((r113_z / $(192)) % $(288)) * $(64)) + (((r113_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r113_value = r113_value.to(tl.bfloat16)
    r113_value = tl.broadcast_to(r113_value, [$(442368)])
    tl.store($(vcat_10_ptr) + r113_z, r113_value)
    r114_row = tl.arange($(288))
    r114_col = tl.arange($(512))
    r114_lane = tl.arange($(2048))
    r114_left = tl.load($(vcat_9_ptr) + ((((((((((r114_row[:, None] * $(2048)) + r114_lane[None, :]) / $(2048)) % $(288)) * $(2048)) + (((r114_row[:, None] * $(2048)) + r114_lane[None, :]) % $(2048))) / $(256)) % $(8)) * $(73728)) + (((((r114_row[:, None] * $(2048)) + r114_lane[None, :]) / $(2048)) % $(288)) * $(256)) + (((((((r114_row[:, None] * $(2048)) + r114_lane[None, :]) / $(2048)) % $(288)) * $(2048)) + (((r114_row[:, None] * $(2048)) + r114_lane[None, :]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r114_right = tl.load($(vto_copy_5_ptr) + (((r114_lane[:, None] % $(2048)) * $(512)) + (r114_col[None, :] % $(512))), dtype=tl.bfloat16)
    r114_value = tl.dot(r114_left, r114_right)
    r114_value = r114_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r114_row[:, None] * $(512)) + r114_col[None, :]), r114_value)
    r115_z = tl.arange($(147456))
    r115_value = tl.load($(vmm_6_ptr) + ((((r115_z / $(512)) % $(288)) * $(512)) + (r115_z % $(512))), dtype=tl.bfloat16)
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(147456)])
    tl.store($(vto_copy_26_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(147456))
    r116_value = (tl.load($(vto_copy_26_ptr) + ((((r116_z / $(512)) % $(288)) * $(512)) + (r116_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_4_ptr) + (r116_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(147456)])
    tl.store($(vmul_31_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(147456))
    r117_value = (tl.load($(vmul_31_ptr) + ((((r117_z / $(512)) % $(288)) * $(512)) + (r117_z % $(512))), dtype=tl.float32) * tl.load($(vto_copy_20_ptr) + ((((r117_z / $(512)) % $(288)) * $(512)) + (r117_z % $(512))), dtype=tl.float32))
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(147456)])
    tl.store($(vmul_32_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(147456))
    r118_value = (tl.load($(vmul_31_ptr) + ((((r118_z / $(512)) % $(288)) * $(512)) + (r118_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r118_z / $(512)) % $(288)), dtype=tl.float32))
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(147456)])
    tl.store($(vmul_33_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(288))
    r119_lane = tl.arange($(512))
    r119_value = tl.sum(tl.load($(vmul_32_ptr) + ((r119_z[:, None] * $(512)) + r119_lane[None, :]), dtype=tl.float32), axis=1)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(288)])
    tl.store($(vsum_2_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(288))
    r120_value = ((tl.load($(vrsqrt_1_ptr) + r120_z, dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + r120_z, dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + r120_z, dtype=tl.float32)))
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(288)])
    tl.store($(vpow_4_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(288))
    r121_value = (tl.load($(vsum_2_ptr) + r121_z, dtype=tl.float32) * -0.5)
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(288)])
    tl.store($(vmul_34_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(288))
    r122_value = (tl.load($(vmul_34_ptr) + r122_z, dtype=tl.float32) * tl.load($(vpow_4_ptr) + r122_z, dtype=tl.float32))
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(288)])
    tl.store($(vmul_35_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(147456))
    r123_value = (tl.load($(vmul_35_ptr) + ((r123_z / $(512)) % $(288)), dtype=tl.float32) / 512.0)
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(147456)])
    tl.store($(vdiv_1_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(147456))
    r124_value = ((tl.load($(vto_copy_20_ptr) + ((((r124_z / $(512)) % $(288)) * $(512)) + (r124_z % $(512))), dtype=tl.float32)))
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(147456)])
    tl.store($(vpow_5_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(147456))
    r125_value = (tl.load($(vpow_5_ptr) + ((((r125_z / $(512)) % $(288)) * $(512)) + (r125_z % $(512))), dtype=tl.float32) * 2.0)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(147456)])
    tl.store($(vmul_36_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(147456))
    r126_value = (tl.load($(vdiv_1_ptr) + ((((r126_z / $(512)) % $(288)) * $(512)) + (r126_z % $(512))), dtype=tl.float32) * tl.load($(vmul_36_ptr) + ((((r126_z / $(512)) % $(288)) * $(512)) + (r126_z % $(512))), dtype=tl.float32))
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(147456)])
    tl.store($(vmul_37_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(147456))
    r127_value = (tl.load($(vmul_33_ptr) + ((((r127_z / $(512)) % $(288)) * $(512)) + (r127_z % $(512))), dtype=tl.float32) + tl.load($(vmul_37_ptr) + ((((r127_z / $(512)) % $(288)) * $(512)) + (r127_z % $(512))), dtype=tl.float32))
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(147456)])
    tl.store($(vadd_13_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(147456))
    r128_value = tl.load($(vadd_13_ptr) + ((((r128_z / $(512)) % $(288)) * $(512)) + (r128_z % $(512))), dtype=tl.float32)
    r128_value = r128_value.to(tl.bfloat16)
    r128_value = tl.broadcast_to(r128_value, [$(147456)])
    tl.store($(vto_copy_27_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(165888))
    r129_value = tl.where((r129_z % $(576)) < $(512), tl.load($(vto_copy_27_ptr) + ((((r129_z / $(576)) % $(288)) * $(512)) + (((r129_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(vadd_11_ptr) + ((((((((r129_z / $(576)) % $(288)) * $(64)) + ((r129_z % $(576)) - $(512))) / $(64)) % $(288)) * $(64)) + (((((r129_z / $(576)) % $(288)) * $(64)) + ((r129_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r129_value = r129_value.to(tl.bfloat16)
    r129_value = tl.broadcast_to(r129_value, [$(165888)])
    tl.store($(vcat_11_ptr) + r129_z, r129_value)
    r130_row = tl.arange($(288))
    r130_col = tl.arange($(512))
    r130_lane = tl.arange($(576))
    r130_left = tl.load($(vcat_11_ptr) + ((((((r130_row[:, None] * $(576)) + r130_lane[None, :]) / $(576)) % $(288)) * $(576)) + (((r130_row[:, None] * $(576)) + r130_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r130_right = tl.load($(vto_copy_3_ptr) + (((r130_lane[:, None] % $(576)) * $(512)) + (r130_col[None, :] % $(512))), dtype=tl.bfloat16)
    r130_value = tl.dot(r130_left, r130_right)
    r130_value = r130_value.to(tl.bfloat16)
    tl.store($(vmm_7_ptr) + ((r130_row[:, None] * $(512)) + r130_col[None, :]), r130_value)
    r131_row = tl.arange($(288))
    r131_col = tl.arange($(256))
    r131_lane = tl.arange($(1536))
    r131_left = tl.load($(vcat_10_ptr) + ((((((((((r131_row[:, None] * $(1536)) + r131_lane[None, :]) / $(1536)) % $(288)) * $(1536)) + (((r131_row[:, None] * $(1536)) + r131_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(55296)) + (((((r131_row[:, None] * $(1536)) + r131_lane[None, :]) / $(1536)) % $(288)) * $(192)) + (((((((r131_row[:, None] * $(1536)) + r131_lane[None, :]) / $(1536)) % $(288)) * $(1536)) + (((r131_row[:, None] * $(1536)) + r131_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r131_right = tl.load($(vto_copy_2_ptr) + (((r131_lane[:, None] % $(1536)) * $(256)) + (r131_col[None, :] % $(256))), dtype=tl.bfloat16)
    r131_value = tl.dot(r131_left, r131_right)
    r131_value = r131_value.to(tl.bfloat16)
    tl.store($(vmm_8_ptr) + ((r131_row[:, None] * $(256)) + r131_col[None, :]), r131_value)
    r132_z = tl.arange($(73728))
    r132_value = tl.load($(vmm_8_ptr) + ((((r132_z / $(256)) % $(288)) * $(256)) + (r132_z % $(256))), dtype=tl.bfloat16)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(73728)])
    tl.store($(vto_copy_28_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(73728))
    r133_value = (tl.load($(vto_copy_28_ptr) + ((((r133_z / $(256)) % $(288)) * $(256)) + (r133_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_1_ptr) + (r133_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(73728)])
    tl.store($(vmul_38_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(73728))
    r134_value = (tl.load($(vmul_38_ptr) + ((((r134_z / $(256)) % $(288)) * $(256)) + (r134_z % $(256))), dtype=tl.float32) * tl.load($(vto_copy_18_ptr) + ((((r134_z / $(256)) % $(288)) * $(256)) + (r134_z % $(256))), dtype=tl.float32))
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(73728)])
    tl.store($(vmul_39_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(73728))
    r135_value = (tl.load($(vmul_38_ptr) + ((((r135_z / $(256)) % $(288)) * $(256)) + (r135_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((r135_z / $(256)) % $(288)), dtype=tl.float32))
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(73728)])
    tl.store($(vmul_40_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(288))
    r136_lane = tl.arange($(256))
    r136_value = tl.sum(tl.load($(vmul_39_ptr) + ((r136_z[:, None] * $(256)) + r136_lane[None, :]), dtype=tl.float32), axis=1)
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(288)])
    tl.store($(vsum_3_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(288))
    r137_value = ((tl.load($(vrsqrt_ptr) + r137_z, dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + r137_z, dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + r137_z, dtype=tl.float32)))
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(288)])
    tl.store($(vpow_6_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(288))
    r138_value = (tl.load($(vsum_3_ptr) + r138_z, dtype=tl.float32) * -0.5)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(288)])
    tl.store($(vmul_41_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(288))
    r139_value = (tl.load($(vmul_41_ptr) + r139_z, dtype=tl.float32) * tl.load($(vpow_6_ptr) + r139_z, dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(288)])
    tl.store($(vmul_42_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(73728))
    r140_value = (tl.load($(vmul_42_ptr) + ((r140_z / $(256)) % $(288)), dtype=tl.float32) / 256.0)
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(73728)])
    tl.store($(vdiv_2_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(73728))
    r141_value = ((tl.load($(vto_copy_18_ptr) + ((((r141_z / $(256)) % $(288)) * $(256)) + (r141_z % $(256))), dtype=tl.float32)))
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(73728)])
    tl.store($(vpow_7_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(73728))
    r142_value = (tl.load($(vpow_7_ptr) + ((((r142_z / $(256)) % $(288)) * $(256)) + (r142_z % $(256))), dtype=tl.float32) * 2.0)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(73728)])
    tl.store($(vmul_43_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(73728))
    r143_value = (tl.load($(vdiv_2_ptr) + ((((r143_z / $(256)) % $(288)) * $(256)) + (r143_z % $(256))), dtype=tl.float32) * tl.load($(vmul_43_ptr) + ((((r143_z / $(256)) % $(288)) * $(256)) + (r143_z % $(256))), dtype=tl.float32))
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(73728)])
    tl.store($(vmul_44_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(73728))
    r144_value = (tl.load($(vmul_40_ptr) + ((((r144_z / $(256)) % $(288)) * $(256)) + (r144_z % $(256))), dtype=tl.float32) + tl.load($(vmul_44_ptr) + ((((r144_z / $(256)) % $(288)) * $(256)) + (r144_z % $(256))), dtype=tl.float32))
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(73728)])
    tl.store($(vadd_14_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(73728))
    r145_value = tl.load($(vadd_14_ptr) + ((((r145_z / $(256)) % $(288)) * $(256)) + (r145_z % $(256))), dtype=tl.float32)
    r145_value = r145_value.to(tl.bfloat16)
    r145_value = tl.broadcast_to(r145_value, [$(73728)])
    tl.store($(vto_copy_29_ptr) + r145_z, r145_value)
    r146_row = tl.arange($(288))
    r146_col = tl.arange($(512))
    r146_lane = tl.arange($(256))
    r146_left = tl.load($(vto_copy_29_ptr) + ((((((r146_row[:, None] * $(256)) + r146_lane[None, :]) / $(256)) % $(288)) * $(256)) + (((r146_row[:, None] * $(256)) + r146_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r146_right = tl.load($(vto_copy_ptr) + (((r146_lane[:, None] % $(256)) * $(512)) + (r146_col[None, :] % $(512))), dtype=tl.bfloat16)
    r146_value = tl.dot(r146_left, r146_right)
    r146_value = r146_value.to(tl.bfloat16)
    tl.store($(vmm_9_ptr) + ((r146_row[:, None] * $(512)) + r146_col[None, :]), r146_value)
    r147_z = tl.arange($(147456))
    r147_value = (tl.load($(vmm_7_ptr) + ((((r147_z / $(512)) % $(288)) * $(512)) + (r147_z % $(512))), dtype=tl.bfloat16) + tl.load($(vmm_9_ptr) + ((((r147_z / $(512)) % $(288)) * $(512)) + (r147_z % $(512))), dtype=tl.bfloat16))
    r147_value = r147_value.to(tl.bfloat16)
    r147_value = tl.broadcast_to(r147_value, [$(147456)])
    tl.store($(vadd_15_ptr) + r147_z, r147_value)
    z = tl.arange($(147456))
    value = tl.load($(vadd_15_ptr) + ((((z / $(512)) % $(288)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(147456)])
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
  let vsum_1_ptr : RegionName := "vsum_1"
  let vneg_ptr : RegionName := "vneg"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vpow_5_ptr : RegionName := "vpow_5"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vpow_6_ptr : RegionName := "vpow_6"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vpow_7_ptr : RegionName := "vpow_7"
  let vmul_43_ptr : RegionName := "vmul_43"
  let vmul_44_ptr : RegionName := "vmul_44"
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
  let candidate_vindex_put_ptr : RegionName := "candidate_vindex_put"
  let candidate_vmm_22_ptr : RegionName := "candidate_vmm_22"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vcat_18_ptr : RegionName := "candidate_vcat_18"
  let candidate_vcat_19_ptr : RegionName := "candidate_vcat_19"
  let candidate_vto_copy_67_ptr : RegionName := "candidate_vto_copy_67"
  let candidate_vto_copy_68_ptr : RegionName := "candidate_vto_copy_68"
  let candidate_vbmm_29_ptr : RegionName := "candidate_vbmm_29"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vsub_24_ptr : RegionName := "candidate_vsub_24"
  let candidate_vgt_9_ptr : RegionName := "candidate_vgt_9"
  let candidate_vmasked_fill_9_ptr : RegionName := "candidate_vmasked_fill_9"
  let candidate_vsoftmax_9_ptr : RegionName := "candidate_vsoftmax_9"
  let candidate_vto_copy_69_ptr : RegionName := "candidate_vto_copy_69"
  let candidate_vbmm_30_ptr : RegionName := "candidate_vbmm_30"
  let candidate_vlogsumexp_9_ptr : RegionName := "candidate_vlogsumexp_9"
  let candidate_vcat_20_ptr : RegionName := "candidate_vcat_20"
  let candidate_vcat_21_ptr : RegionName := "candidate_vcat_21"
  let candidate_vto_copy_70_ptr : RegionName := "candidate_vto_copy_70"
  let candidate_vto_copy_71_ptr : RegionName := "candidate_vto_copy_71"
  let candidate_vto_copy_72_ptr : RegionName := "candidate_vto_copy_72"
  let candidate_vto_copy_73_ptr : RegionName := "candidate_vto_copy_73"
  let candidate_vto_copy_74_ptr : RegionName := "candidate_vto_copy_74"
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
  let candidate_vbmm_33_ptr : RegionName := "candidate_vbmm_33"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vto_copy_75_ptr : RegionName := "candidate_vto_copy_75"
  let candidate_vbmm_34_ptr : RegionName := "candidate_vbmm_34"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vto_copy_76_ptr : RegionName := "candidate_vto_copy_76"
  let candidate_vbmm_35_ptr : RegionName := "candidate_vbmm_35"
  let candidate_vto_copy_77_ptr : RegionName := "candidate_vto_copy_77"
  let candidate_vcat_22_ptr : RegionName := "candidate_vcat_22"
  let candidate_vcat_23_ptr : RegionName := "candidate_vcat_23"
  let candidate_vcat_24_ptr : RegionName := "candidate_vcat_24"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vbmm_36_ptr : RegionName := "candidate_vbmm_36"
  let candidate_vbmm_37_ptr : RegionName := "candidate_vbmm_37"
  let candidate_vadd_26_ptr : RegionName := "candidate_vadd_26"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vbmm_40_ptr : RegionName := "candidate_vbmm_40"
  let candidate_vbmm_41_ptr : RegionName := "candidate_vbmm_41"
  let candidate_vcat_26_ptr : RegionName := "candidate_vcat_26"
  let candidate_vcat_27_ptr : RegionName := "candidate_vcat_27"
  let candidate_vto_copy_78_ptr : RegionName := "candidate_vto_copy_78"
  let candidate_vto_copy_79_ptr : RegionName := "candidate_vto_copy_79"
  let candidate_vbmm_42_ptr : RegionName := "candidate_vbmm_42"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vadd_27_ptr : RegionName := "candidate_vadd_27"
  let candidate_vsub_28_ptr : RegionName := "candidate_vsub_28"
  let candidate_vgt_11_ptr : RegionName := "candidate_vgt_11"
  let candidate_vmasked_fill_11_ptr : RegionName := "candidate_vmasked_fill_11"
  let candidate_vsoftmax_10_ptr : RegionName := "candidate_vsoftmax_10"
  let candidate_vto_copy_80_ptr : RegionName := "candidate_vto_copy_80"
  let candidate_vbmm_43_ptr : RegionName := "candidate_vbmm_43"
  let candidate_vlogsumexp_10_ptr : RegionName := "candidate_vlogsumexp_10"
  let candidate_vcat_28_ptr : RegionName := "candidate_vcat_28"
  let candidate_vcat_29_ptr : RegionName := "candidate_vcat_29"
  let candidate_vto_copy_81_ptr : RegionName := "candidate_vto_copy_81"
  let candidate_vto_copy_82_ptr : RegionName := "candidate_vto_copy_82"
  let candidate_vto_copy_83_ptr : RegionName := "candidate_vto_copy_83"
  let candidate_vto_copy_84_ptr : RegionName := "candidate_vto_copy_84"
  let candidate_vto_copy_85_ptr : RegionName := "candidate_vto_copy_85"
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
  let candidate_vbmm_46_ptr : RegionName := "candidate_vbmm_46"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vto_copy_86_ptr : RegionName := "candidate_vto_copy_86"
  let candidate_vbmm_47_ptr : RegionName := "candidate_vbmm_47"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vto_copy_87_ptr : RegionName := "candidate_vto_copy_87"
  let candidate_vbmm_48_ptr : RegionName := "candidate_vbmm_48"
  let candidate_vto_copy_88_ptr : RegionName := "candidate_vto_copy_88"
  let candidate_vcat_30_ptr : RegionName := "candidate_vcat_30"
  let candidate_vcat_31_ptr : RegionName := "candidate_vcat_31"
  let candidate_vcat_32_ptr : RegionName := "candidate_vcat_32"
  let candidate_vcat_33_ptr : RegionName := "candidate_vcat_33"
  let candidate_vcat_34_ptr : RegionName := "candidate_vcat_34"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vbmm_49_ptr : RegionName := "candidate_vbmm_49"
  let candidate_vbmm_50_ptr : RegionName := "candidate_vbmm_50"
  let candidate_vadd_29_ptr : RegionName := "candidate_vadd_29"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vadd_30_ptr : RegionName := "candidate_vadd_30"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_31_ptr : RegionName := "candidate_vadd_31"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vadd_32_ptr : RegionName := "candidate_vadd_32"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vadd_33_ptr : RegionName := "candidate_vadd_33"
  let candidate_vbmm_53_ptr : RegionName := "candidate_vbmm_53"
  let candidate_vbmm_54_ptr : RegionName := "candidate_vbmm_54"
  let candidate_vcat_36_ptr : RegionName := "candidate_vcat_36"
  let candidate_vcat_37_ptr : RegionName := "candidate_vcat_37"
  let candidate_vto_copy_89_ptr : RegionName := "candidate_vto_copy_89"
  let candidate_vto_copy_90_ptr : RegionName := "candidate_vto_copy_90"
  let candidate_vbmm_55_ptr : RegionName := "candidate_vbmm_55"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vadd_34_ptr : RegionName := "candidate_vadd_34"
  let candidate_vsub_32_ptr : RegionName := "candidate_vsub_32"
  let candidate_vgt_13_ptr : RegionName := "candidate_vgt_13"
  let candidate_vmasked_fill_13_ptr : RegionName := "candidate_vmasked_fill_13"
  let candidate_vsoftmax_11_ptr : RegionName := "candidate_vsoftmax_11"
  let candidate_vto_copy_91_ptr : RegionName := "candidate_vto_copy_91"
  let candidate_vbmm_56_ptr : RegionName := "candidate_vbmm_56"
  let candidate_vlogsumexp_11_ptr : RegionName := "candidate_vlogsumexp_11"
  let candidate_vcat_38_ptr : RegionName := "candidate_vcat_38"
  let candidate_vcat_39_ptr : RegionName := "candidate_vcat_39"
  let candidate_vto_copy_92_ptr : RegionName := "candidate_vto_copy_92"
  let candidate_vto_copy_93_ptr : RegionName := "candidate_vto_copy_93"
  let candidate_vto_copy_94_ptr : RegionName := "candidate_vto_copy_94"
  let candidate_vto_copy_95_ptr : RegionName := "candidate_vto_copy_95"
  let candidate_vto_copy_96_ptr : RegionName := "candidate_vto_copy_96"
  let candidate_vbmm_57_ptr : RegionName := "candidate_vbmm_57"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vadd_35_ptr : RegionName := "candidate_vadd_35"
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
  let candidate_vbmm_59_ptr : RegionName := "candidate_vbmm_59"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vto_copy_97_ptr : RegionName := "candidate_vto_copy_97"
  let candidate_vbmm_60_ptr : RegionName := "candidate_vbmm_60"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vto_copy_98_ptr : RegionName := "candidate_vto_copy_98"
  let candidate_vbmm_61_ptr : RegionName := "candidate_vbmm_61"
  let candidate_vto_copy_99_ptr : RegionName := "candidate_vto_copy_99"
  let candidate_vcat_40_ptr : RegionName := "candidate_vcat_40"
  let candidate_vcat_41_ptr : RegionName := "candidate_vcat_41"
  let candidate_vcat_42_ptr : RegionName := "candidate_vcat_42"
  let candidate_vcat_43_ptr : RegionName := "candidate_vcat_43"
  let candidate_vcat_44_ptr : RegionName := "candidate_vcat_44"
  let candidate_vsum_6_ptr : RegionName := "candidate_vsum_6"
  let candidate_vbmm_62_ptr : RegionName := "candidate_vbmm_62"
  let candidate_vbmm_63_ptr : RegionName := "candidate_vbmm_63"
  let candidate_vadd_36_ptr : RegionName := "candidate_vadd_36"
  let candidate_vslice_backward_8_ptr : RegionName := "candidate_vslice_backward_8"
  let candidate_vadd_37_ptr : RegionName := "candidate_vadd_37"
  let candidate_vslice_backward_9_ptr : RegionName := "candidate_vslice_backward_9"
  let candidate_vadd_38_ptr : RegionName := "candidate_vadd_38"
  let candidate_vslice_backward_10_ptr : RegionName := "candidate_vslice_backward_10"
  let candidate_vadd_39_ptr : RegionName := "candidate_vadd_39"
  let candidate_vslice_backward_11_ptr : RegionName := "candidate_vslice_backward_11"
  let candidate_vadd_40_ptr : RegionName := "candidate_vadd_40"
  let candidate_vbmm_66_ptr : RegionName := "candidate_vbmm_66"
  let candidate_vbmm_67_ptr : RegionName := "candidate_vbmm_67"
  let candidate_vcat_46_ptr : RegionName := "candidate_vcat_46"
  let candidate_vcat_47_ptr : RegionName := "candidate_vcat_47"
  let candidate_vto_copy_100_ptr : RegionName := "candidate_vto_copy_100"
  let candidate_vto_copy_101_ptr : RegionName := "candidate_vto_copy_101"
  let candidate_vbmm_68_ptr : RegionName := "candidate_vbmm_68"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vadd_41_ptr : RegionName := "candidate_vadd_41"
  let candidate_vsub_36_ptr : RegionName := "candidate_vsub_36"
  let candidate_vgt_15_ptr : RegionName := "candidate_vgt_15"
  let candidate_vmasked_fill_15_ptr : RegionName := "candidate_vmasked_fill_15"
  let candidate_vsoftmax_12_ptr : RegionName := "candidate_vsoftmax_12"
  let candidate_vto_copy_102_ptr : RegionName := "candidate_vto_copy_102"
  let candidate_vbmm_69_ptr : RegionName := "candidate_vbmm_69"
  let candidate_vlogsumexp_12_ptr : RegionName := "candidate_vlogsumexp_12"
  let candidate_vcat_48_ptr : RegionName := "candidate_vcat_48"
  let candidate_vcat_49_ptr : RegionName := "candidate_vcat_49"
  let candidate_vto_copy_103_ptr : RegionName := "candidate_vto_copy_103"
  let candidate_vto_copy_104_ptr : RegionName := "candidate_vto_copy_104"
  let candidate_vto_copy_105_ptr : RegionName := "candidate_vto_copy_105"
  let candidate_vto_copy_106_ptr : RegionName := "candidate_vto_copy_106"
  let candidate_vto_copy_107_ptr : RegionName := "candidate_vto_copy_107"
  let candidate_vbmm_70_ptr : RegionName := "candidate_vbmm_70"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vadd_42_ptr : RegionName := "candidate_vadd_42"
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
  let candidate_vbmm_72_ptr : RegionName := "candidate_vbmm_72"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vto_copy_108_ptr : RegionName := "candidate_vto_copy_108"
  let candidate_vbmm_73_ptr : RegionName := "candidate_vbmm_73"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vto_copy_109_ptr : RegionName := "candidate_vto_copy_109"
  let candidate_vbmm_74_ptr : RegionName := "candidate_vbmm_74"
  let candidate_vto_copy_110_ptr : RegionName := "candidate_vto_copy_110"
  let candidate_vcat_50_ptr : RegionName := "candidate_vcat_50"
  let candidate_vcat_51_ptr : RegionName := "candidate_vcat_51"
  let candidate_vcat_52_ptr : RegionName := "candidate_vcat_52"
  let candidate_vsum_8_ptr : RegionName := "candidate_vsum_8"
  let candidate_vbmm_75_ptr : RegionName := "candidate_vbmm_75"
  let candidate_vbmm_76_ptr : RegionName := "candidate_vbmm_76"
  let candidate_vadd_43_ptr : RegionName := "candidate_vadd_43"
  let candidate_vslice_backward_12_ptr : RegionName := "candidate_vslice_backward_12"
  let candidate_vadd_44_ptr : RegionName := "candidate_vadd_44"
  let candidate_vslice_backward_13_ptr : RegionName := "candidate_vslice_backward_13"
  let candidate_vadd_45_ptr : RegionName := "candidate_vadd_45"
  let candidate_vslice_backward_14_ptr : RegionName := "candidate_vslice_backward_14"
  let candidate_vadd_46_ptr : RegionName := "candidate_vadd_46"
  let candidate_vslice_backward_15_ptr : RegionName := "candidate_vslice_backward_15"
  let candidate_vadd_47_ptr : RegionName := "candidate_vadd_47"
  let candidate_vbmm_79_ptr : RegionName := "candidate_vbmm_79"
  let candidate_vbmm_80_ptr : RegionName := "candidate_vbmm_80"
  let candidate_vcat_54_ptr : RegionName := "candidate_vcat_54"
  let candidate_vcat_55_ptr : RegionName := "candidate_vcat_55"
  let candidate_vto_copy_111_ptr : RegionName := "candidate_vto_copy_111"
  let candidate_vto_copy_112_ptr : RegionName := "candidate_vto_copy_112"
  let candidate_vbmm_81_ptr : RegionName := "candidate_vbmm_81"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vadd_48_ptr : RegionName := "candidate_vadd_48"
  let candidate_vsub_40_ptr : RegionName := "candidate_vsub_40"
  let candidate_vgt_17_ptr : RegionName := "candidate_vgt_17"
  let candidate_vmasked_fill_17_ptr : RegionName := "candidate_vmasked_fill_17"
  let candidate_vsoftmax_13_ptr : RegionName := "candidate_vsoftmax_13"
  let candidate_vto_copy_113_ptr : RegionName := "candidate_vto_copy_113"
  let candidate_vbmm_82_ptr : RegionName := "candidate_vbmm_82"
  let candidate_vlogsumexp_13_ptr : RegionName := "candidate_vlogsumexp_13"
  let candidate_vcat_56_ptr : RegionName := "candidate_vcat_56"
  let candidate_vcat_57_ptr : RegionName := "candidate_vcat_57"
  let candidate_vto_copy_114_ptr : RegionName := "candidate_vto_copy_114"
  let candidate_vto_copy_115_ptr : RegionName := "candidate_vto_copy_115"
  let candidate_vto_copy_116_ptr : RegionName := "candidate_vto_copy_116"
  let candidate_vto_copy_117_ptr : RegionName := "candidate_vto_copy_117"
  let candidate_vto_copy_118_ptr : RegionName := "candidate_vto_copy_118"
  let candidate_vbmm_83_ptr : RegionName := "candidate_vbmm_83"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vadd_49_ptr : RegionName := "candidate_vadd_49"
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
  let candidate_vbmm_85_ptr : RegionName := "candidate_vbmm_85"
  let candidate_vmul_58_ptr : RegionName := "candidate_vmul_58"
  let candidate_vto_copy_119_ptr : RegionName := "candidate_vto_copy_119"
  let candidate_vbmm_86_ptr : RegionName := "candidate_vbmm_86"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vto_copy_120_ptr : RegionName := "candidate_vto_copy_120"
  let candidate_vbmm_87_ptr : RegionName := "candidate_vbmm_87"
  let candidate_vto_copy_121_ptr : RegionName := "candidate_vto_copy_121"
  let candidate_vcat_58_ptr : RegionName := "candidate_vcat_58"
  let candidate_vcat_59_ptr : RegionName := "candidate_vcat_59"
  let candidate_vcat_60_ptr : RegionName := "candidate_vcat_60"
  let candidate_vcat_61_ptr : RegionName := "candidate_vcat_61"
  let candidate_vcat_62_ptr : RegionName := "candidate_vcat_62"
  let candidate_vsum_10_ptr : RegionName := "candidate_vsum_10"
  let candidate_vbmm_88_ptr : RegionName := "candidate_vbmm_88"
  let candidate_vbmm_89_ptr : RegionName := "candidate_vbmm_89"
  let candidate_vadd_50_ptr : RegionName := "candidate_vadd_50"
  let candidate_vslice_backward_16_ptr : RegionName := "candidate_vslice_backward_16"
  let candidate_vadd_51_ptr : RegionName := "candidate_vadd_51"
  let candidate_vslice_backward_17_ptr : RegionName := "candidate_vslice_backward_17"
  let candidate_vadd_52_ptr : RegionName := "candidate_vadd_52"
  let candidate_vslice_backward_18_ptr : RegionName := "candidate_vslice_backward_18"
  let candidate_vadd_53_ptr : RegionName := "candidate_vadd_53"
  let candidate_vslice_backward_19_ptr : RegionName := "candidate_vslice_backward_19"
  let candidate_vadd_54_ptr : RegionName := "candidate_vadd_54"
  let candidate_vbmm_92_ptr : RegionName := "candidate_vbmm_92"
  let candidate_vbmm_93_ptr : RegionName := "candidate_vbmm_93"
  let candidate_vcat_64_ptr : RegionName := "candidate_vcat_64"
  let candidate_vcat_65_ptr : RegionName := "candidate_vcat_65"
  let candidate_vto_copy_122_ptr : RegionName := "candidate_vto_copy_122"
  let candidate_vto_copy_123_ptr : RegionName := "candidate_vto_copy_123"
  let candidate_vbmm_94_ptr : RegionName := "candidate_vbmm_94"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vadd_55_ptr : RegionName := "candidate_vadd_55"
  let candidate_vsub_44_ptr : RegionName := "candidate_vsub_44"
  let candidate_vgt_19_ptr : RegionName := "candidate_vgt_19"
  let candidate_vmasked_fill_19_ptr : RegionName := "candidate_vmasked_fill_19"
  let candidate_vsoftmax_14_ptr : RegionName := "candidate_vsoftmax_14"
  let candidate_vto_copy_124_ptr : RegionName := "candidate_vto_copy_124"
  let candidate_vbmm_95_ptr : RegionName := "candidate_vbmm_95"
  let candidate_vlogsumexp_14_ptr : RegionName := "candidate_vlogsumexp_14"
  let candidate_vcat_66_ptr : RegionName := "candidate_vcat_66"
  let candidate_vcat_67_ptr : RegionName := "candidate_vcat_67"
  let candidate_vto_copy_125_ptr : RegionName := "candidate_vto_copy_125"
  let candidate_vto_copy_126_ptr : RegionName := "candidate_vto_copy_126"
  let candidate_vto_copy_127_ptr : RegionName := "candidate_vto_copy_127"
  let candidate_vto_copy_128_ptr : RegionName := "candidate_vto_copy_128"
  let candidate_vto_copy_129_ptr : RegionName := "candidate_vto_copy_129"
  let candidate_vbmm_96_ptr : RegionName := "candidate_vbmm_96"
  let candidate_vmul_61_ptr : RegionName := "candidate_vmul_61"
  let candidate_vadd_56_ptr : RegionName := "candidate_vadd_56"
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
  let candidate_vbmm_98_ptr : RegionName := "candidate_vbmm_98"
  let candidate_vmul_64_ptr : RegionName := "candidate_vmul_64"
  let candidate_vto_copy_130_ptr : RegionName := "candidate_vto_copy_130"
  let candidate_vbmm_99_ptr : RegionName := "candidate_vbmm_99"
  let candidate_vmul_65_ptr : RegionName := "candidate_vmul_65"
  let candidate_vto_copy_131_ptr : RegionName := "candidate_vto_copy_131"
  let candidate_vbmm_100_ptr : RegionName := "candidate_vbmm_100"
  let candidate_vto_copy_132_ptr : RegionName := "candidate_vto_copy_132"
  let candidate_vcat_68_ptr : RegionName := "candidate_vcat_68"
  let candidate_vcat_69_ptr : RegionName := "candidate_vcat_69"
  let candidate_vcat_70_ptr : RegionName := "candidate_vcat_70"
  let candidate_vcat_71_ptr : RegionName := "candidate_vcat_71"
  let candidate_vcat_72_ptr : RegionName := "candidate_vcat_72"
  let candidate_vsum_12_ptr : RegionName := "candidate_vsum_12"
  let candidate_vbmm_101_ptr : RegionName := "candidate_vbmm_101"
  let candidate_vbmm_102_ptr : RegionName := "candidate_vbmm_102"
  let candidate_vadd_57_ptr : RegionName := "candidate_vadd_57"
  let candidate_vslice_backward_20_ptr : RegionName := "candidate_vslice_backward_20"
  let candidate_vadd_58_ptr : RegionName := "candidate_vadd_58"
  let candidate_vslice_backward_21_ptr : RegionName := "candidate_vslice_backward_21"
  let candidate_vadd_59_ptr : RegionName := "candidate_vadd_59"
  let candidate_vslice_backward_22_ptr : RegionName := "candidate_vslice_backward_22"
  let candidate_vadd_60_ptr : RegionName := "candidate_vadd_60"
  let candidate_vslice_backward_23_ptr : RegionName := "candidate_vslice_backward_23"
  let candidate_vadd_61_ptr : RegionName := "candidate_vadd_61"
  let candidate_vbmm_105_ptr : RegionName := "candidate_vbmm_105"
  let candidate_vbmm_106_ptr : RegionName := "candidate_vbmm_106"
  let candidate_vcat_74_ptr : RegionName := "candidate_vcat_74"
  let candidate_vcat_75_ptr : RegionName := "candidate_vcat_75"
  let candidate_vto_copy_133_ptr : RegionName := "candidate_vto_copy_133"
  let candidate_vto_copy_134_ptr : RegionName := "candidate_vto_copy_134"
  let candidate_vbmm_107_ptr : RegionName := "candidate_vbmm_107"
  let candidate_vmul_66_ptr : RegionName := "candidate_vmul_66"
  let candidate_vadd_62_ptr : RegionName := "candidate_vadd_62"
  let candidate_vsub_48_ptr : RegionName := "candidate_vsub_48"
  let candidate_vgt_21_ptr : RegionName := "candidate_vgt_21"
  let candidate_vmasked_fill_21_ptr : RegionName := "candidate_vmasked_fill_21"
  let candidate_vsoftmax_15_ptr : RegionName := "candidate_vsoftmax_15"
  let candidate_vto_copy_135_ptr : RegionName := "candidate_vto_copy_135"
  let candidate_vbmm_108_ptr : RegionName := "candidate_vbmm_108"
  let candidate_vlogsumexp_15_ptr : RegionName := "candidate_vlogsumexp_15"
  let candidate_vcat_76_ptr : RegionName := "candidate_vcat_76"
  let candidate_vcat_77_ptr : RegionName := "candidate_vcat_77"
  let candidate_vto_copy_136_ptr : RegionName := "candidate_vto_copy_136"
  let candidate_vto_copy_137_ptr : RegionName := "candidate_vto_copy_137"
  let candidate_vto_copy_138_ptr : RegionName := "candidate_vto_copy_138"
  let candidate_vto_copy_139_ptr : RegionName := "candidate_vto_copy_139"
  let candidate_vto_copy_140_ptr : RegionName := "candidate_vto_copy_140"
  let candidate_vbmm_109_ptr : RegionName := "candidate_vbmm_109"
  let candidate_vmul_67_ptr : RegionName := "candidate_vmul_67"
  let candidate_vadd_63_ptr : RegionName := "candidate_vadd_63"
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
  let candidate_vbmm_111_ptr : RegionName := "candidate_vbmm_111"
  let candidate_vmul_70_ptr : RegionName := "candidate_vmul_70"
  let candidate_vto_copy_141_ptr : RegionName := "candidate_vto_copy_141"
  let candidate_vbmm_112_ptr : RegionName := "candidate_vbmm_112"
  let candidate_vmul_71_ptr : RegionName := "candidate_vmul_71"
  let candidate_vto_copy_142_ptr : RegionName := "candidate_vto_copy_142"
  let candidate_vbmm_113_ptr : RegionName := "candidate_vbmm_113"
  let candidate_vto_copy_143_ptr : RegionName := "candidate_vto_copy_143"
  let candidate_vcat_78_ptr : RegionName := "candidate_vcat_78"
  let candidate_vcat_79_ptr : RegionName := "candidate_vcat_79"
  let candidate_vcat_80_ptr : RegionName := "candidate_vcat_80"
  let candidate_vsum_14_ptr : RegionName := "candidate_vsum_14"
  let candidate_vbmm_114_ptr : RegionName := "candidate_vbmm_114"
  let candidate_vbmm_115_ptr : RegionName := "candidate_vbmm_115"
  let candidate_vadd_64_ptr : RegionName := "candidate_vadd_64"
  let candidate_vslice_backward_24_ptr : RegionName := "candidate_vslice_backward_24"
  let candidate_vadd_65_ptr : RegionName := "candidate_vadd_65"
  let candidate_vslice_backward_25_ptr : RegionName := "candidate_vslice_backward_25"
  let candidate_vadd_66_ptr : RegionName := "candidate_vadd_66"
  let candidate_vslice_backward_26_ptr : RegionName := "candidate_vslice_backward_26"
  let candidate_vadd_67_ptr : RegionName := "candidate_vadd_67"
  let candidate_vslice_backward_27_ptr : RegionName := "candidate_vslice_backward_27"
  let candidate_vadd_68_ptr : RegionName := "candidate_vadd_68"
  let candidate_vbmm_118_ptr : RegionName := "candidate_vbmm_118"
  let candidate_vbmm_119_ptr : RegionName := "candidate_vbmm_119"
  let candidate_vcat_82_ptr : RegionName := "candidate_vcat_82"
  let candidate_vcat_83_ptr : RegionName := "candidate_vcat_83"
  let candidate_vto_copy_144_ptr : RegionName := "candidate_vto_copy_144"
  let candidate_vto_copy_145_ptr : RegionName := "candidate_vto_copy_145"
  let candidate_vbmm_120_ptr : RegionName := "candidate_vbmm_120"
  let candidate_vmul_72_ptr : RegionName := "candidate_vmul_72"
  let candidate_vadd_69_ptr : RegionName := "candidate_vadd_69"
  let candidate_vsub_52_ptr : RegionName := "candidate_vsub_52"
  let candidate_vgt_23_ptr : RegionName := "candidate_vgt_23"
  let candidate_vmasked_fill_23_ptr : RegionName := "candidate_vmasked_fill_23"
  let candidate_vsoftmax_16_ptr : RegionName := "candidate_vsoftmax_16"
  let candidate_vto_copy_146_ptr : RegionName := "candidate_vto_copy_146"
  let candidate_vbmm_121_ptr : RegionName := "candidate_vbmm_121"
  let candidate_vlogsumexp_16_ptr : RegionName := "candidate_vlogsumexp_16"
  let candidate_vcat_84_ptr : RegionName := "candidate_vcat_84"
  let candidate_vcat_85_ptr : RegionName := "candidate_vcat_85"
  let candidate_vto_copy_147_ptr : RegionName := "candidate_vto_copy_147"
  let candidate_vto_copy_148_ptr : RegionName := "candidate_vto_copy_148"
  let candidate_vto_copy_149_ptr : RegionName := "candidate_vto_copy_149"
  let candidate_vto_copy_150_ptr : RegionName := "candidate_vto_copy_150"
  let candidate_vto_copy_151_ptr : RegionName := "candidate_vto_copy_151"
  let candidate_vbmm_122_ptr : RegionName := "candidate_vbmm_122"
  let candidate_vmul_73_ptr : RegionName := "candidate_vmul_73"
  let candidate_vadd_70_ptr : RegionName := "candidate_vadd_70"
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
  let candidate_vbmm_124_ptr : RegionName := "candidate_vbmm_124"
  let candidate_vmul_76_ptr : RegionName := "candidate_vmul_76"
  let candidate_vto_copy_152_ptr : RegionName := "candidate_vto_copy_152"
  let candidate_vbmm_125_ptr : RegionName := "candidate_vbmm_125"
  let candidate_vmul_77_ptr : RegionName := "candidate_vmul_77"
  let candidate_vto_copy_153_ptr : RegionName := "candidate_vto_copy_153"
  let candidate_vbmm_126_ptr : RegionName := "candidate_vbmm_126"
  let candidate_vto_copy_154_ptr : RegionName := "candidate_vto_copy_154"
  let candidate_vcat_86_ptr : RegionName := "candidate_vcat_86"
  let candidate_vcat_87_ptr : RegionName := "candidate_vcat_87"
  let candidate_vcat_88_ptr : RegionName := "candidate_vcat_88"
  let candidate_vcat_89_ptr : RegionName := "candidate_vcat_89"
  let candidate_vcat_90_ptr : RegionName := "candidate_vcat_90"
  let candidate_vsum_16_ptr : RegionName := "candidate_vsum_16"
  let candidate_vbmm_127_ptr : RegionName := "candidate_vbmm_127"
  let candidate_vbmm_128_ptr : RegionName := "candidate_vbmm_128"
  let candidate_vadd_71_ptr : RegionName := "candidate_vadd_71"
  let candidate_vslice_backward_28_ptr : RegionName := "candidate_vslice_backward_28"
  let candidate_vadd_72_ptr : RegionName := "candidate_vadd_72"
  let candidate_vslice_backward_29_ptr : RegionName := "candidate_vslice_backward_29"
  let candidate_vadd_73_ptr : RegionName := "candidate_vadd_73"
  let candidate_vslice_backward_30_ptr : RegionName := "candidate_vslice_backward_30"
  let candidate_vadd_74_ptr : RegionName := "candidate_vadd_74"
  let candidate_vslice_backward_31_ptr : RegionName := "candidate_vslice_backward_31"
  let candidate_vadd_75_ptr : RegionName := "candidate_vadd_75"
  let candidate_vbmm_131_ptr : RegionName := "candidate_vbmm_131"
  let candidate_vbmm_132_ptr : RegionName := "candidate_vbmm_132"
  let candidate_vcat_92_ptr : RegionName := "candidate_vcat_92"
  let candidate_vcat_93_ptr : RegionName := "candidate_vcat_93"
  let candidate_vto_copy_155_ptr : RegionName := "candidate_vto_copy_155"
  let candidate_vto_copy_156_ptr : RegionName := "candidate_vto_copy_156"
  let candidate_vbmm_133_ptr : RegionName := "candidate_vbmm_133"
  let candidate_vmul_78_ptr : RegionName := "candidate_vmul_78"
  let candidate_vadd_76_ptr : RegionName := "candidate_vadd_76"
  let candidate_vsub_56_ptr : RegionName := "candidate_vsub_56"
  let candidate_vgt_25_ptr : RegionName := "candidate_vgt_25"
  let candidate_vmasked_fill_25_ptr : RegionName := "candidate_vmasked_fill_25"
  let candidate_vsoftmax_17_ptr : RegionName := "candidate_vsoftmax_17"
  let candidate_vto_copy_157_ptr : RegionName := "candidate_vto_copy_157"
  let candidate_vbmm_134_ptr : RegionName := "candidate_vbmm_134"
  let candidate_vlogsumexp_17_ptr : RegionName := "candidate_vlogsumexp_17"
  let candidate_vcat_94_ptr : RegionName := "candidate_vcat_94"
  let candidate_vcat_95_ptr : RegionName := "candidate_vcat_95"
  let candidate_vto_copy_158_ptr : RegionName := "candidate_vto_copy_158"
  let candidate_vto_copy_159_ptr : RegionName := "candidate_vto_copy_159"
  let candidate_vto_copy_160_ptr : RegionName := "candidate_vto_copy_160"
  let candidate_vto_copy_161_ptr : RegionName := "candidate_vto_copy_161"
  let candidate_vto_copy_162_ptr : RegionName := "candidate_vto_copy_162"
  let candidate_vbmm_135_ptr : RegionName := "candidate_vbmm_135"
  let candidate_vmul_79_ptr : RegionName := "candidate_vmul_79"
  let candidate_vadd_77_ptr : RegionName := "candidate_vadd_77"
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
  let candidate_vbmm_137_ptr : RegionName := "candidate_vbmm_137"
  let candidate_vmul_82_ptr : RegionName := "candidate_vmul_82"
  let candidate_vto_copy_163_ptr : RegionName := "candidate_vto_copy_163"
  let candidate_vbmm_138_ptr : RegionName := "candidate_vbmm_138"
  let candidate_vmul_83_ptr : RegionName := "candidate_vmul_83"
  let candidate_vto_copy_164_ptr : RegionName := "candidate_vto_copy_164"
  let candidate_vbmm_139_ptr : RegionName := "candidate_vbmm_139"
  let candidate_vto_copy_165_ptr : RegionName := "candidate_vto_copy_165"
  let candidate_vcat_96_ptr : RegionName := "candidate_vcat_96"
  let candidate_vcat_97_ptr : RegionName := "candidate_vcat_97"
  let candidate_vcat_98_ptr : RegionName := "candidate_vcat_98"
  let candidate_vcat_99_ptr : RegionName := "candidate_vcat_99"
  let candidate_vcat_100_ptr : RegionName := "candidate_vcat_100"
  let candidate_vsum_18_ptr : RegionName := "candidate_vsum_18"
  let candidate_vbmm_140_ptr : RegionName := "candidate_vbmm_140"
  let candidate_vbmm_141_ptr : RegionName := "candidate_vbmm_141"
  let candidate_vadd_78_ptr : RegionName := "candidate_vadd_78"
  let candidate_vslice_backward_32_ptr : RegionName := "candidate_vslice_backward_32"
  let candidate_vadd_79_ptr : RegionName := "candidate_vadd_79"
  let candidate_vslice_backward_33_ptr : RegionName := "candidate_vslice_backward_33"
  let candidate_vadd_80_ptr : RegionName := "candidate_vadd_80"
  let candidate_vslice_backward_34_ptr : RegionName := "candidate_vslice_backward_34"
  let candidate_vadd_81_ptr : RegionName := "candidate_vadd_81"
  let candidate_vslice_backward_35_ptr : RegionName := "candidate_vslice_backward_35"
  let candidate_vadd_82_ptr : RegionName := "candidate_vadd_82"
  let candidate_vcat_102_ptr : RegionName := "candidate_vcat_102"
  let candidate_vmul_84_ptr : RegionName := "candidate_vmul_84"
  let candidate_vmul_85_ptr : RegionName := "candidate_vmul_85"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vmul_86_ptr : RegionName := "candidate_vmul_86"
  let candidate_vadd_83_ptr : RegionName := "candidate_vadd_83"
  let candidate_vmul_87_ptr : RegionName := "candidate_vmul_87"
  let candidate_vadd_84_ptr : RegionName := "candidate_vadd_84"
  let candidate_vmul_88_ptr : RegionName := "candidate_vmul_88"
  let candidate_vmul_89_ptr : RegionName := "candidate_vmul_89"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vmul_90_ptr : RegionName := "candidate_vmul_90"
  let candidate_vadd_85_ptr : RegionName := "candidate_vadd_85"
  let candidate_vmul_91_ptr : RegionName := "candidate_vmul_91"
  let candidate_vadd_86_ptr : RegionName := "candidate_vadd_86"
  let candidate_vslice_backward_36_ptr : RegionName := "candidate_vslice_backward_36"
  let candidate_vslice_backward_37_ptr : RegionName := "candidate_vslice_backward_37"
  let candidate_vadd_87_ptr : RegionName := "candidate_vadd_87"
  let candidate_vslice_backward_38_ptr : RegionName := "candidate_vslice_backward_38"
  let candidate_vslice_backward_39_ptr : RegionName := "candidate_vslice_backward_39"
  let candidate_vadd_88_ptr : RegionName := "candidate_vadd_88"
  let candidate_vcat_103_ptr : RegionName := "candidate_vcat_103"
  let candidate_vto_copy_166_ptr : RegionName := "candidate_vto_copy_166"
  let candidate_vmul_92_ptr : RegionName := "candidate_vmul_92"
  let candidate_vmul_93_ptr : RegionName := "candidate_vmul_93"
  let candidate_vmul_94_ptr : RegionName := "candidate_vmul_94"
  let candidate_vsum_19_ptr : RegionName := "candidate_vsum_19"
  let candidate_vpow_4_ptr : RegionName := "candidate_vpow_4"
  let candidate_vmul_95_ptr : RegionName := "candidate_vmul_95"
  let candidate_vmul_96_ptr : RegionName := "candidate_vmul_96"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_vpow_5_ptr : RegionName := "candidate_vpow_5"
  let candidate_vmul_97_ptr : RegionName := "candidate_vmul_97"
  let candidate_vmul_98_ptr : RegionName := "candidate_vmul_98"
  let candidate_vadd_89_ptr : RegionName := "candidate_vadd_89"
  let candidate_vto_copy_167_ptr : RegionName := "candidate_vto_copy_167"
  let candidate_vcat_104_ptr : RegionName := "candidate_vcat_104"
  let candidate_vmm_23_ptr : RegionName := "candidate_vmm_23"
  let candidate_vmm_24_ptr : RegionName := "candidate_vmm_24"
  let candidate_vto_copy_168_ptr : RegionName := "candidate_vto_copy_168"
  let candidate_vmul_99_ptr : RegionName := "candidate_vmul_99"
  let candidate_vmul_100_ptr : RegionName := "candidate_vmul_100"
  let candidate_vmul_101_ptr : RegionName := "candidate_vmul_101"
  let candidate_vsum_20_ptr : RegionName := "candidate_vsum_20"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_102_ptr : RegionName := "candidate_vmul_102"
  let candidate_vmul_103_ptr : RegionName := "candidate_vmul_103"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vpow_7_ptr : RegionName := "candidate_vpow_7"
  let candidate_vmul_104_ptr : RegionName := "candidate_vmul_104"
  let candidate_vmul_105_ptr : RegionName := "candidate_vmul_105"
  let candidate_vadd_90_ptr : RegionName := "candidate_vadd_90"
  let candidate_vto_copy_169_ptr : RegionName := "candidate_vto_copy_169"
  let candidate_vmm_25_ptr : RegionName := "candidate_vmm_25"
  let candidate_vadd_91_ptr : RegionName := "candidate_vadd_91"
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
      r66_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r66_lane[:, None] * $(1024)) + r66_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r66_lane[:, None] * $(1024)) + r66_col[None, :]) % $(128)) * $(512)) + ((((r66_lane[:, None] * $(1024)) + r66_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r66_value = tl.dot(r66_left, r66_right)
      r66_value = r66_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_27_ptr) + ((r66_group * $(98304)) + (r66_row[:, None] * $(1024)) + r66_col[None, :]), r66_value)
    }
    r67_row = tl.arange($(96))
    r67_col = tl.arange($(1024))
    r67_lane = tl.arange($(512))
    for r67_group in range($(1)) {
      r67_left = tl.load($(candidate_vcat_7_ptr) + ((((((r67_row[:, None] * $(512)) + r67_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r67_row[:, None] * $(512)) + r67_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r67_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r67_lane[:, None] * $(1024)) + r67_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r67_lane[:, None] * $(1024)) + r67_col[None, :]) % $(128)) * $(512)) + ((((r67_lane[:, None] * $(1024)) + r67_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_67_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(147456))
    r71_value = tl.load($(candidate_vcat_19_ptr) + ((((r71_z / $(192)) % $(96)) * $(1536)) + (((r71_z / $(18432)) % $(8)) * $(192)) + (r71_z % $(192))), dtype=tl.bfloat16)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(147456)])
    tl.store($(candidate_vto_copy_68_ptr) + r71_z, r71_value)
    r72_row = tl.arange($(32))
    r72_col = tl.arange($(96))
    r72_lane = tl.arange($(192))
    for r72_group in range($(8)) {
      r72_left = tl.load($(candidate_vto_copy_67_ptr) + (((((((r72_group % $(8)) * $(6144)) + (r72_row[:, None] * $(192)) + r72_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r72_group % $(8)) * $(6144)) + (r72_row[:, None] * $(192)) + r72_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r72_group % $(8)) * $(6144)) + (r72_row[:, None] * $(192)) + r72_lane[None, :]) % $(192))), dtype=tl.float32)
      r72_right = tl.load($(candidate_vto_copy_68_ptr) + (((((((r72_group % $(8)) * $(18432)) + (r72_lane[:, None] * $(96)) + r72_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r72_group % $(8)) * $(18432)) + (r72_lane[:, None] * $(96)) + r72_col[None, :]) % $(96)) * $(192)) + (((((r72_group % $(8)) * $(18432)) + (r72_lane[:, None] * $(96)) + r72_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_69_ptr) + r79_z, r79_value)
    r80_row = tl.arange($(32))
    r80_col = tl.arange($(128))
    r80_lane = tl.arange($(96))
    for r80_group in range($(8)) {
      r80_left = tl.load($(candidate_vto_copy_69_ptr) + (((((((r80_group % $(8)) * $(3072)) + (r80_row[:, None] * $(96)) + r80_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r80_group % $(8)) * $(3072)) + (r80_row[:, None] * $(96)) + r80_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r80_group % $(8)) * $(3072)) + (r80_row[:, None] * $(96)) + r80_lane[None, :]) % $(96))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_70_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(147456))
    r85_value = tl.load($(candidate_vcat_19_ptr) + ((((r85_z / $(192)) % $(96)) * $(1536)) + (((r85_z / $(18432)) % $(8)) * $(192)) + (r85_z % $(192))), dtype=tl.bfloat16)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(147456)])
    tl.store($(candidate_vto_copy_71_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(98304))
    r86_value = tl.load($(candidate_vbmm_28_ptr) + ((((r86_z / $(128)) % $(96)) * $(1024)) + (((r86_z / $(12288)) % $(8)) * $(128)) + (r86_z % $(128))), dtype=tl.bfloat16)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(98304)])
    tl.store($(candidate_vto_copy_72_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(32768))
    r87_value = tl.load($(candidate_vmm_22_ptr) + ((((r87_z / $(128)) % $(32)) * $(1024)) + (((r87_z / $(4096)) % $(8)) * $(128)) + (r87_z % $(128)) + $(262144)), dtype=tl.bfloat16)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(32768)])
    tl.store($(candidate_vto_copy_73_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(32768))
    r88_value = tl.load($(candidate_vcat_20_ptr) + ((((r88_z / $(128)) % $(32)) * $(1024)) + (((r88_z / $(4096)) % $(8)) * $(128)) + (r88_z % $(128))), dtype=tl.bfloat16)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(32768)])
    tl.store($(candidate_vto_copy_74_ptr) + r88_z, r88_value)
    r89_row = tl.arange($(32))
    r89_col = tl.arange($(96))
    r89_lane = tl.arange($(192))
    for r89_group in range($(8)) {
      r89_left = tl.load($(candidate_vto_copy_70_ptr) + (((((((r89_group % $(8)) * $(6144)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r89_group % $(8)) * $(6144)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r89_group % $(8)) * $(6144)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) % $(192))), dtype=tl.float32)
      r89_right = tl.load($(candidate_vto_copy_71_ptr) + (((((((r89_group % $(8)) * $(18432)) + (r89_lane[:, None] * $(96)) + r89_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r89_group % $(8)) * $(18432)) + (r89_lane[:, None] * $(96)) + r89_col[None, :]) % $(96)) * $(192)) + (((((r89_group % $(8)) * $(18432)) + (r89_lane[:, None] * $(96)) + r89_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
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
    r97_value = (tl.load($(candidate_vto_copy_73_ptr) + ((((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(128)) % $(32)) * $(128)) + (r97_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_74_ptr) + ((((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(128)) % $(32)) * $(128)) + (r97_z % $(128))), dtype=tl.float32))
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
      r99_left = tl.load($(candidate_vto_copy_73_ptr) + (((((((r99_group % $(8)) * $(4096)) + (r99_row[:, None] * $(128)) + r99_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r99_group % $(8)) * $(4096)) + (r99_row[:, None] * $(128)) + r99_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r99_group % $(8)) * $(4096)) + (r99_row[:, None] * $(128)) + r99_lane[None, :]) % $(128))), dtype=tl.float32)
      r99_right = tl.load($(candidate_vto_copy_72_ptr) + (((((((r99_group % $(8)) * $(12288)) + (r99_lane[:, None] * $(96)) + r99_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r99_group % $(8)) * $(12288)) + (r99_lane[:, None] * $(96)) + r99_col[None, :]) % $(96)) * $(128)) + (((((r99_group % $(8)) * $(12288)) + (r99_lane[:, None] * $(96)) + r99_col[None, :]) / $(96)) % $(128))), dtype=tl.float32)
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
    r102_row = tl.arange($(32))
    r102_col = tl.arange($(192))
    r102_lane = tl.arange($(96))
    for r102_group in range($(8)) {
      r102_left = tl.load($(candidate_vmul_33_ptr) + (((((((r102_group % $(8)) * $(3072)) + (r102_row[:, None] * $(96)) + r102_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r102_group % $(8)) * $(3072)) + (r102_row[:, None] * $(96)) + r102_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r102_group % $(8)) * $(3072)) + (r102_row[:, None] * $(96)) + r102_lane[None, :]) % $(96))), dtype=tl.float32)
      r102_right = tl.load($(candidate_vto_copy_71_ptr) + (((((((r102_group % $(8)) * $(18432)) + (r102_lane[:, None] * $(192)) + r102_col[None, :]) / $(18432)) % $(8)) * $(18432)) + ((((((r102_group % $(8)) * $(18432)) + (r102_lane[:, None] * $(192)) + r102_col[None, :]) / $(192)) % $(96)) * $(192)) + ((((r102_group % $(8)) * $(18432)) + (r102_lane[:, None] * $(192)) + r102_col[None, :]) % $(192))), dtype=tl.float32)
      r102_value = tl.dot(r102_left, r102_right)
      r102_value = r102_value.to(tl.float32)
      tl.store($(candidate_vbmm_33_ptr) + ((r102_group * $(6144)) + (r102_row[:, None] * $(192)) + r102_col[None, :]), r102_value)
    }
    r103_z = tl.arange($(49152))
    r103_value = (tl.load($(candidate_vbmm_33_ptr) + ((((r103_z / $(6144)) % $(8)) * $(6144)) + (((r103_z / $(192)) % $(32)) * $(192)) + (r103_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(49152)])
    tl.store($(candidate_vmul_34_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(49152))
    r104_value = tl.load($(candidate_vmul_34_ptr) + ((((r104_z / $(192)) % $(8)) * $(6144)) + (((r104_z / $(1536)) % $(32)) * $(192)) + (r104_z % $(192))), dtype=tl.float32)
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(49152)])
    tl.store($(candidate_vto_copy_75_ptr) + r104_z, r104_value)
    r105_row = tl.arange($(96))
    r105_col = tl.arange($(192))
    r105_lane = tl.arange($(32))
    for r105_group in range($(8)) {
      r105_left = tl.load($(candidate_vmul_33_ptr) + (((((((r105_group % $(8)) * $(3072)) + (r105_row[:, None] * $(32)) + r105_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r105_group % $(8)) * $(3072)) + (r105_row[:, None] * $(32)) + r105_lane[None, :]) % $(32)) * $(96)) + (((((r105_group % $(8)) * $(3072)) + (r105_row[:, None] * $(32)) + r105_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r105_right = tl.load($(candidate_vto_copy_70_ptr) + (((((((r105_group % $(8)) * $(6144)) + (r105_lane[:, None] * $(192)) + r105_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r105_group % $(8)) * $(6144)) + (r105_lane[:, None] * $(192)) + r105_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r105_group % $(8)) * $(6144)) + (r105_lane[:, None] * $(192)) + r105_col[None, :]) % $(192))), dtype=tl.float32)
      r105_value = tl.dot(r105_left, r105_right)
      r105_value = r105_value.to(tl.float32)
      tl.store($(candidate_vbmm_34_ptr) + ((r105_group * $(18432)) + (r105_row[:, None] * $(192)) + r105_col[None, :]), r105_value)
    }
    r106_z = tl.arange($(147456))
    r106_value = (tl.load($(candidate_vbmm_34_ptr) + ((((r106_z / $(18432)) % $(8)) * $(18432)) + (((r106_z / $(192)) % $(96)) * $(192)) + (r106_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(147456)])
    tl.store($(candidate_vmul_35_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(147456))
    r107_value = tl.load($(candidate_vmul_35_ptr) + ((((r107_z / $(192)) % $(8)) * $(18432)) + (((r107_z / $(1536)) % $(96)) * $(192)) + (r107_z % $(192))), dtype=tl.float32)
    r107_value = r107_value.to(tl.bfloat16)
    r107_value = tl.broadcast_to(r107_value, [$(147456)])
    tl.store($(candidate_vto_copy_76_ptr) + r107_z, r107_value)
    r108_row = tl.arange($(96))
    r108_col = tl.arange($(128))
    r108_lane = tl.arange($(32))
    for r108_group in range($(8)) {
      r108_left = tl.load($(candidate_vexp_ptr) + (((((((r108_group % $(8)) * $(3072)) + (r108_row[:, None] * $(32)) + r108_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r108_group % $(8)) * $(3072)) + (r108_row[:, None] * $(32)) + r108_lane[None, :]) % $(32)) * $(96)) + (((((r108_group % $(8)) * $(3072)) + (r108_row[:, None] * $(32)) + r108_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r108_right = tl.load($(candidate_vto_copy_73_ptr) + (((((((r108_group % $(8)) * $(4096)) + (r108_lane[:, None] * $(128)) + r108_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r108_group % $(8)) * $(4096)) + (r108_lane[:, None] * $(128)) + r108_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r108_group % $(8)) * $(4096)) + (r108_lane[:, None] * $(128)) + r108_col[None, :]) % $(128))), dtype=tl.float32)
      r108_value = tl.dot(r108_left, r108_right)
      r108_value = r108_value.to(tl.float32)
      tl.store($(candidate_vbmm_35_ptr) + ((r108_group * $(12288)) + (r108_row[:, None] * $(128)) + r108_col[None, :]), r108_value)
    }
    r109_z = tl.arange($(98304))
    r109_value = tl.load($(candidate_vbmm_35_ptr) + ((((r109_z / $(128)) % $(8)) * $(12288)) + (((r109_z / $(1024)) % $(96)) * $(128)) + (r109_z % $(128))), dtype=tl.float32)
    r109_value = r109_value.to(tl.bfloat16)
    r109_value = tl.broadcast_to(r109_value, [$(98304)])
    tl.store($(candidate_vto_copy_77_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(49152))
    r110_value = tl.load($(candidate_vto_copy_75_ptr) + ((((((r110_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(1536)) + (((r110_z / $(192)) % $(8)) * $(192)) + (r110_z % $(192))), dtype=tl.bfloat16)
    r110_value = r110_value.to(tl.bfloat16)
    r110_value = tl.broadcast_to(r110_value, [$(49152)])
    tl.store($(candidate_vcat_22_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(147456))
    r111_value = tl.load($(candidate_vto_copy_76_ptr) + ((((((r111_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r111_z / $(192)) % $(8)) * $(192)) + (r111_z % $(192))), dtype=tl.bfloat16)
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(147456)])
    tl.store($(candidate_vcat_23_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(98304))
    r112_value = tl.load($(candidate_vto_copy_77_ptr) + ((((((r112_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r112_z / $(128)) % $(8)) * $(128)) + (r112_z % $(128))), dtype=tl.bfloat16)
    r112_value = r112_value.to(tl.bfloat16)
    r112_value = tl.broadcast_to(r112_value, [$(98304)])
    tl.store($(candidate_vcat_24_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(6144))
    r113_lane = tl.arange($(8))
    r113_value = tl.sum(tl.load($(candidate_vcat_23_ptr) + ((((r113_z[:, None] / $(64)) % $(96)) * $(1536)) + (r113_lane[None, :] * $(192)) + (r113_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r113_value = r113_value.to(tl.bfloat16)
    r113_value = tl.broadcast_to(r113_value, [$(6144)])
    tl.store($(candidate_vsum_2_ptr) + r113_z, r113_value)
    r114_row = tl.arange($(96))
    r114_col = tl.arange($(512))
    r114_lane = tl.arange($(1024))
    for r114_group in range($(1)) {
      r114_left = tl.load($(candidate_vcat_23_ptr) + ((((((r114_row[:, None] * $(1024)) + r114_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + ((((r114_row[:, None] * $(1024)) + r114_lane[None, :]) % $(8)) * $(192)) + ((((r114_row[:, None] * $(1024)) + r114_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r114_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r114_lane[:, None] * $(512)) + r114_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r114_lane[:, None] * $(512)) + r114_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r114_lane[:, None] * $(512)) + r114_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r114_value = tl.dot(r114_left, r114_right)
      r114_value = r114_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_36_ptr) + ((r114_group * $(49152)) + (r114_row[:, None] * $(512)) + r114_col[None, :]), r114_value)
    }
    r115_row = tl.arange($(96))
    r115_col = tl.arange($(512))
    r115_lane = tl.arange($(1024))
    for r115_group in range($(1)) {
      r115_left = tl.load($(candidate_vcat_24_ptr) + ((((((r115_row[:, None] * $(1024)) + r115_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + ((((r115_row[:, None] * $(1024)) + r115_lane[None, :]) % $(8)) * $(128)) + ((((r115_row[:, None] * $(1024)) + r115_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r115_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r115_lane[:, None] * $(512)) + r115_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r115_lane[:, None] * $(512)) + r115_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r115_lane[:, None] * $(512)) + r115_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r115_value = tl.dot(r115_left, r115_right)
      r115_value = r115_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_37_ptr) + ((r115_group * $(49152)) + (r115_row[:, None] * $(512)) + r115_col[None, :]), r115_value)
    }
    r116_z = tl.arange($(49152))
    r116_value = (tl.load($(candidate_vbmm_36_ptr) + ((((r116_z / $(512)) % $(96)) * $(512)) + (r116_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_37_ptr) + ((((r116_z / $(512)) % $(96)) * $(512)) + (r116_z % $(512))), dtype=tl.bfloat16))
    r116_value = r116_value.to(tl.bfloat16)
    r116_value = tl.broadcast_to(r116_value, [$(49152)])
    tl.store($(candidate_vadd_26_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(147456))
    r117_value = tl.where(((((r117_z / $(512)) % $(288)) >= $(192)) & (((r117_z / $(512)) % $(288)) < $(288)) & ((((r117_z / $(512)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vadd_26_ptr) + ((((((r117_z / $(512)) % $(288)) - $(192)) % $(96)) * $(512)) + (r117_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r117_value = r117_value.to(tl.bfloat16)
    r117_value = tl.broadcast_to(r117_value, [$(147456)])
    tl.store($(candidate_vslice_backward_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(18432))
    r118_value = tl.where(((((r118_z / $(64)) % $(288)) >= $(192)) & (((r118_z / $(64)) % $(288)) < $(288)) & ((((r118_z / $(64)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vsum_2_ptr) + ((((((r118_z / $(64)) % $(288)) - $(192)) % $(96)) * $(64)) + (r118_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r118_value = r118_value.to(tl.bfloat16)
    r118_value = tl.broadcast_to(r118_value, [$(18432)])
    tl.store($(candidate_vslice_backward_1_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(294912))
    r119_value = tl.where(((((r119_z / $(1024)) % $(288)) >= $(256)) & (((r119_z / $(1024)) % $(288)) < $(288)) & ((((r119_z / $(1024)) % $(288)) - $(256)) % $(1) == $(0))), tl.load($(candidate_vcat_22_ptr) + ((((((r119_z / $(1024)) % $(288)) - $(256)) % $(32)) * $(1536)) + (((r119_z / $(128)) % $(8)) * $(192)) + (r119_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r119_value = r119_value.to(tl.bfloat16)
    r119_value = tl.broadcast_to(r119_value, [$(294912)])
    tl.store($(candidate_vslice_backward_2_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(147456))
    r120_value = tl.where(((((r120_z / $(512)) % $(288)) >= $(256)) & (((r120_z / $(512)) % $(288)) < $(288)) & ((((r120_z / $(512)) % $(288)) - $(256)) % $(1) == $(0))), tl.load($(candidate_vcat_22_ptr) + ((((((r120_z / $(512)) % $(288)) - $(256)) % $(32)) * $(1536)) + (((r120_z / $(64)) % $(8)) * $(192)) + (r120_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r120_value = r120_value.to(tl.bfloat16)
    r120_value = tl.broadcast_to(r120_value, [$(147456)])
    tl.store($(candidate_vslice_backward_3_ptr) + r120_z, r120_value)
    r121_row = tl.arange($(96))
    r121_col = tl.arange($(1024))
    r121_lane = tl.arange($(512))
    for r121_group in range($(1)) {
      r121_left = tl.load($(candidate_vcat_7_ptr) + ((((((r121_row[:, None] * $(512)) + r121_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r121_row[:, None] * $(512)) + r121_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r121_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r121_lane[:, None] * $(1024)) + r121_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r121_lane[:, None] * $(1024)) + r121_col[None, :]) % $(128)) * $(512)) + ((((r121_lane[:, None] * $(1024)) + r121_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r121_value = tl.dot(r121_left, r121_right)
      r121_value = r121_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_40_ptr) + ((r121_group * $(98304)) + (r121_row[:, None] * $(1024)) + r121_col[None, :]), r121_value)
    }
    r122_row = tl.arange($(96))
    r122_col = tl.arange($(1024))
    r122_lane = tl.arange($(512))
    for r122_group in range($(1)) {
      r122_left = tl.load($(candidate_vcat_7_ptr) + ((((((r122_row[:, None] * $(512)) + r122_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r122_row[:, None] * $(512)) + r122_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r122_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r122_lane[:, None] * $(1024)) + r122_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r122_lane[:, None] * $(1024)) + r122_col[None, :]) % $(128)) * $(512)) + ((((r122_lane[:, None] * $(1024)) + r122_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r122_value = tl.dot(r122_left, r122_right)
      r122_value = r122_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_41_ptr) + ((r122_group * $(98304)) + (r122_row[:, None] * $(1024)) + r122_col[None, :]), r122_value)
    }
    r123_z = tl.arange($(49152))
    r123_value = tl.where((r123_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r123_z / $(1536)) % $(32)) * $(1024)) + (((r123_z / $(192)) % $(8)) * $(128)) + ((r123_z % $(192)) - $(0)) + $(229376)) / $(1024)) % $(288)) * $(1536)) + (((((((r123_z / $(1536)) % $(32)) * $(1024)) + (((r123_z / $(192)) % $(8)) * $(128)) + ((r123_z % $(192)) - $(0)) + $(229376)) / $(128)) % $(8)) * $(192)) + (((((r123_z / $(1536)) % $(32)) * $(1024)) + (((r123_z / $(192)) % $(8)) * $(128)) + ((r123_z % $(192)) - $(0)) + $(229376)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r123_z / $(1536)) % $(32)) * $(512)) + (((r123_z / $(192)) % $(8)) * $(64)) + ((r123_z % $(192)) - $(128)) + $(114688)) / $(64)) % $(8)) * $(18432)) + (((((((r123_z / $(1536)) % $(32)) * $(512)) + (((r123_z / $(192)) % $(8)) * $(64)) + ((r123_z % $(192)) - $(128)) + $(114688)) / $(512)) % $(288)) * $(64)) + (((((r123_z / $(1536)) % $(32)) * $(512)) + (((r123_z / $(192)) % $(8)) * $(64)) + ((r123_z % $(192)) - $(128)) + $(114688)) % $(64))), dtype=tl.bfloat16))
    r123_value = r123_value.to(tl.bfloat16)
    r123_value = tl.broadcast_to(r123_value, [$(49152)])
    tl.store($(candidate_vcat_26_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(147456))
    r124_value = tl.where((r124_z % $(192)) < $(128), tl.load($(candidate_vbmm_40_ptr) + ((((((((r124_z / $(1536)) % $(96)) * $(1024)) + (((r124_z / $(192)) % $(8)) * $(128)) + ((r124_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r124_z / $(1536)) % $(96)) * $(1024)) + (((r124_z / $(192)) % $(8)) * $(128)) + ((r124_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r124_z / $(1536)) % $(96)) * $(1024)) + (((r124_z / $(192)) % $(8)) * $(128)) + ((r124_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r124_z / $(1536)) % $(96)) * $(576)) + ((((r124_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r124_value = r124_value.to(tl.bfloat16)
    r124_value = tl.broadcast_to(r124_value, [$(147456)])
    tl.store($(candidate_vcat_27_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(49152))
    r125_value = tl.load($(candidate_vcat_26_ptr) + ((((r125_z / $(192)) % $(32)) * $(1536)) + (((r125_z / $(6144)) % $(8)) * $(192)) + (r125_z % $(192))), dtype=tl.bfloat16)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(49152)])
    tl.store($(candidate_vto_copy_78_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(98304))
    r126_value = tl.load($(candidate_vcat_27_ptr) + ((((r126_z / $(192)) % $(64)) * $(1536)) + (((r126_z / $(12288)) % $(8)) * $(192)) + (r126_z % $(192))), dtype=tl.bfloat16)
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(98304)])
    tl.store($(candidate_vto_copy_79_ptr) + r126_z, r126_value)
    r127_row = tl.arange($(32))
    r127_col = tl.arange($(64))
    r127_lane = tl.arange($(192))
    for r127_group in range($(8)) {
      r127_left = tl.load($(candidate_vto_copy_78_ptr) + (((((((r127_group % $(8)) * $(6144)) + (r127_row[:, None] * $(192)) + r127_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r127_group % $(8)) * $(6144)) + (r127_row[:, None] * $(192)) + r127_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r127_group % $(8)) * $(6144)) + (r127_row[:, None] * $(192)) + r127_lane[None, :]) % $(192))), dtype=tl.float32)
      r127_right = tl.load($(candidate_vto_copy_79_ptr) + (((((((r127_group % $(8)) * $(12288)) + (r127_lane[:, None] * $(64)) + r127_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r127_group % $(8)) * $(12288)) + (r127_lane[:, None] * $(64)) + r127_col[None, :]) % $(64)) * $(192)) + (((((r127_group % $(8)) * $(12288)) + (r127_lane[:, None] * $(64)) + r127_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r127_value = tl.dot(r127_left, r127_right)
      r127_value = r127_value.to(tl.float32)
      tl.store($(candidate_vbmm_42_ptr) + ((r127_group * $(2048)) + (r127_row[:, None] * $(64)) + r127_col[None, :]), r127_value)
    }
    r128_z = tl.arange($(16384))
    r128_value = (tl.load($(candidate_vbmm_42_ptr) + ((((r128_z / $(2048)) % $(8)) * $(2048)) + (((r128_z / $(64)) % $(32)) * $(64)) + (r128_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(16384)])
    tl.store($(candidate_vmul_36_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(32))
    r129_value = (tl.cast(r129_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r129_value = tl.broadcast_to(r129_value, [$(32)])
    tl.store($(candidate_vadd_27_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(32))
    r130_value = (tl.load($(candidate_vadd_27_ptr) + r130_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r130_value = tl.broadcast_to(r130_value, [$(32)])
    tl.store($(candidate_vsub_28_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(2048))
    r131_value = (tl.cast((r131_z % $(64)), tl.int64) > tl.load($(candidate_vsub_28_ptr) + ((r131_z / $(64)) % $(32)), dtype=tl.int64))
    r131_value = tl.broadcast_to(r131_value, [$(2048)])
    tl.store($(candidate_vgt_11_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(16384))
    r132_value = tl.where(tl.load($(candidate_vgt_11_ptr) + ((((r132_z / $(64)) % $(32)) * $(64)) + (r132_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_36_ptr) + ((((r132_z / $(2048)) % $(8)) * $(2048)) + (((r132_z / $(64)) % $(32)) * $(64)) + (r132_z % $(64))), dtype=tl.float32))
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_11_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(16384))
    r133_lane = tl.arange($(64))
    r133_row_values = tl.load($(candidate_vmasked_fill_11_ptr) + ((((r133_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r133_z[:, None] / $(64)) % $(32)) * $(64)) + (r133_lane[None, :] % $(64))), dtype=tl.float32)
    r133_maximum = tl.max(r133_row_values, axis=1, keep_dims=true)
    r133_weights = tl.exp(r133_row_values - r133_maximum)
    r133_total = tl.sum(r133_weights, axis=1)
    r133_maximum = tl.max(r133_row_values, axis=1)
    r133_value = tl.exp(tl.load($(candidate_vmasked_fill_11_ptr) + ((((r133_z / $(2048)) % $(8)) * $(2048)) + (((r133_z / $(64)) % $(32)) * $(64)) + (r133_z % $(64))), dtype=tl.float32) - r133_maximum) / r133_total
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(16384)])
    tl.store($(candidate_vsoftmax_10_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(16384))
    r134_value = tl.load($(candidate_vsoftmax_10_ptr) + ((((r134_z / $(2048)) % $(8)) * $(2048)) + (((r134_z / $(64)) % $(32)) * $(64)) + (r134_z % $(64))), dtype=tl.float32)
    r134_value = r134_value.to(tl.bfloat16)
    r134_value = tl.broadcast_to(r134_value, [$(16384)])
    tl.store($(candidate_vto_copy_80_ptr) + r134_z, r134_value)
    r135_row = tl.arange($(32))
    r135_col = tl.arange($(128))
    r135_lane = tl.arange($(64))
    for r135_group in range($(8)) {
      r135_left = tl.load($(candidate_vto_copy_80_ptr) + (((((((r135_group % $(8)) * $(2048)) + (r135_row[:, None] * $(64)) + r135_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r135_group % $(8)) * $(2048)) + (r135_row[:, None] * $(64)) + r135_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r135_group % $(8)) * $(2048)) + (r135_row[:, None] * $(64)) + r135_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r135_right = tl.load($(candidate_vbmm_41_ptr) + (((((((r135_group % $(8)) * $(8192)) + (r135_lane[:, None] * $(128)) + r135_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r135_group % $(8)) * $(8192)) + (r135_lane[:, None] * $(128)) + r135_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r135_group % $(8)) * $(8192)) + (r135_lane[:, None] * $(128)) + r135_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r135_value = tl.dot(r135_left, r135_right)
      r135_value = r135_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_43_ptr) + ((r135_group * $(4096)) + (r135_row[:, None] * $(128)) + r135_col[None, :]), r135_value)
    }
    r136_z = tl.arange($(256))
    r136_lane = tl.arange($(64))
    r136_row_values = tl.load($(candidate_vmasked_fill_11_ptr) + ((((r136_z[:, None] / $(32)) % $(8)) * $(2048)) + ((r136_z[:, None] % $(32)) * $(64)) + (r136_lane[None, :] % $(64))), dtype=tl.float32)
    r136_maximum = tl.max(r136_row_values, axis=1, keep_dims=true)
    r136_weights = tl.exp(r136_row_values - r136_maximum)
    r136_total = tl.sum(r136_weights, axis=1)
    r136_maximum = tl.max(r136_row_values, axis=1)
    r136_value = r136_maximum + tl.log(r136_total)
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(256)])
    tl.store($(candidate_vlogsumexp_10_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(32768))
    r137_value = tl.load($(candidate_vbmm_43_ptr) + ((((((((r137_z / $(128)) % $(8)) * $(4096)) + ((((r137_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r137_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r137_z / $(128)) % $(8)) * $(32)) + (((r137_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r137_z % $(128))), dtype=tl.bfloat16)
    r137_value = r137_value.to(tl.bfloat16)
    r137_value = tl.broadcast_to(r137_value, [$(32768)])
    tl.store($(candidate_vcat_28_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(256))
    r138_value = tl.load($(candidate_vlogsumexp_10_ptr) + (((r138_z % $(8)) * $(32)) + ((((r138_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(256)])
    tl.store($(candidate_vcat_29_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(49152))
    r139_value = tl.load($(candidate_vcat_26_ptr) + ((((r139_z / $(192)) % $(32)) * $(1536)) + (((r139_z / $(6144)) % $(8)) * $(192)) + (r139_z % $(192))), dtype=tl.bfloat16)
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(49152)])
    tl.store($(candidate_vto_copy_81_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(98304))
    r140_value = tl.load($(candidate_vcat_27_ptr) + ((((r140_z / $(192)) % $(64)) * $(1536)) + (((r140_z / $(12288)) % $(8)) * $(192)) + (r140_z % $(192))), dtype=tl.bfloat16)
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(98304)])
    tl.store($(candidate_vto_copy_82_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(65536))
    r141_value = tl.load($(candidate_vbmm_41_ptr) + ((((r141_z / $(128)) % $(64)) * $(1024)) + (((r141_z / $(8192)) % $(8)) * $(128)) + (r141_z % $(128))), dtype=tl.bfloat16)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(65536)])
    tl.store($(candidate_vto_copy_83_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(32768))
    r142_value = tl.load($(candidate_vmm_22_ptr) + ((((r142_z / $(128)) % $(32)) * $(1024)) + (((r142_z / $(4096)) % $(8)) * $(128)) + (r142_z % $(128)) + $(229376)), dtype=tl.bfloat16)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(32768)])
    tl.store($(candidate_vto_copy_84_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(32768))
    r143_value = tl.load($(candidate_vcat_28_ptr) + ((((r143_z / $(128)) % $(32)) * $(1024)) + (((r143_z / $(4096)) % $(8)) * $(128)) + (r143_z % $(128))), dtype=tl.bfloat16)
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(32768)])
    tl.store($(candidate_vto_copy_85_ptr) + r143_z, r143_value)
    r144_row = tl.arange($(32))
    r144_col = tl.arange($(64))
    r144_lane = tl.arange($(192))
    for r144_group in range($(8)) {
      r144_left = tl.load($(candidate_vto_copy_81_ptr) + (((((((r144_group % $(8)) * $(6144)) + (r144_row[:, None] * $(192)) + r144_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r144_group % $(8)) * $(6144)) + (r144_row[:, None] * $(192)) + r144_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r144_group % $(8)) * $(6144)) + (r144_row[:, None] * $(192)) + r144_lane[None, :]) % $(192))), dtype=tl.float32)
      r144_right = tl.load($(candidate_vto_copy_82_ptr) + (((((((r144_group % $(8)) * $(12288)) + (r144_lane[:, None] * $(64)) + r144_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r144_group % $(8)) * $(12288)) + (r144_lane[:, None] * $(64)) + r144_col[None, :]) % $(64)) * $(192)) + (((((r144_group % $(8)) * $(12288)) + (r144_lane[:, None] * $(64)) + r144_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r144_value = tl.dot(r144_left, r144_right)
      r144_value = r144_value.to(tl.float32)
      tl.store($(candidate_vbmm_44_ptr) + ((r144_group * $(2048)) + (r144_row[:, None] * $(64)) + r144_col[None, :]), r144_value)
    }
    r145_z = tl.arange($(16384))
    r145_value = (tl.load($(candidate_vbmm_44_ptr) + ((((r145_z / $(2048)) % $(8)) * $(2048)) + (((r145_z / $(64)) % $(32)) * $(64)) + (r145_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(16384)])
    tl.store($(candidate_vmul_37_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(32))
    r146_value = (tl.cast(r146_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r146_value = tl.broadcast_to(r146_value, [$(32)])
    tl.store($(candidate_vadd_28_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(32))
    r147_value = (tl.load($(candidate_vadd_28_ptr) + r147_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r147_value = tl.broadcast_to(r147_value, [$(32)])
    tl.store($(candidate_vsub_29_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(2048))
    r148_value = (tl.cast((r148_z % $(64)), tl.int64) > tl.load($(candidate_vsub_29_ptr) + ((r148_z / $(64)) % $(32)), dtype=tl.int64))
    r148_value = tl.broadcast_to(r148_value, [$(2048)])
    tl.store($(candidate_vgt_12_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(16384))
    r149_value = tl.where(tl.load($(candidate_vgt_12_ptr) + ((((r149_z / $(64)) % $(32)) * $(64)) + (r149_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_37_ptr) + ((((r149_z / $(2048)) % $(8)) * $(2048)) + (((r149_z / $(64)) % $(32)) * $(64)) + (r149_z % $(64))), dtype=tl.float32))
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_12_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(16384))
    r150_value = (tl.load($(candidate_vmasked_fill_12_ptr) + ((((r150_z / $(2048)) % $(8)) * $(2048)) + (((r150_z / $(64)) % $(32)) * $(64)) + (r150_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_29_ptr) + ((((r150_z / $(64)) % $(32)) * $(8)) + ((r150_z / $(2048)) % $(8))), dtype=tl.float32))
    r150_value = r150_value.to(tl.float32)
    r150_value = tl.broadcast_to(r150_value, [$(16384)])
    tl.store($(candidate_vsub_30_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(16384))
    r151_value = tl.exp(tl.load($(candidate_vsub_30_ptr) + ((((r151_z / $(2048)) % $(8)) * $(2048)) + (((r151_z / $(64)) % $(32)) * $(64)) + (r151_z % $(64))), dtype=tl.float32))
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(16384)])
    tl.store($(candidate_vexp_1_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(32768))
    r152_value = (tl.load($(candidate_vto_copy_84_ptr) + ((((r152_z / $(4096)) % $(8)) * $(4096)) + (((r152_z / $(128)) % $(32)) * $(128)) + (r152_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_85_ptr) + ((((r152_z / $(4096)) % $(8)) * $(4096)) + (((r152_z / $(128)) % $(32)) * $(128)) + (r152_z % $(128))), dtype=tl.float32))
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(32768)])
    tl.store($(candidate_vmul_38_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(256))
    r153_lane = tl.arange($(128))
    r153_value = tl.sum(tl.load($(candidate_vmul_38_ptr) + ((((r153_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r153_z[:, None] % $(32)) * $(128)) + r153_lane[None, :]), dtype=tl.float32), axis=1)
    r153_value = r153_value.to(tl.float32)
    r153_value = tl.broadcast_to(r153_value, [$(256)])
    tl.store($(candidate_vsum_3_ptr) + r153_z, r153_value)
    r154_row = tl.arange($(32))
    r154_col = tl.arange($(64))
    r154_lane = tl.arange($(128))
    for r154_group in range($(8)) {
      r154_left = tl.load($(candidate_vto_copy_84_ptr) + (((((((r154_group % $(8)) * $(4096)) + (r154_row[:, None] * $(128)) + r154_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r154_group % $(8)) * $(4096)) + (r154_row[:, None] * $(128)) + r154_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r154_group % $(8)) * $(4096)) + (r154_row[:, None] * $(128)) + r154_lane[None, :]) % $(128))), dtype=tl.float32)
      r154_right = tl.load($(candidate_vto_copy_83_ptr) + (((((((r154_group % $(8)) * $(8192)) + (r154_lane[:, None] * $(64)) + r154_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r154_group % $(8)) * $(8192)) + (r154_lane[:, None] * $(64)) + r154_col[None, :]) % $(64)) * $(128)) + (((((r154_group % $(8)) * $(8192)) + (r154_lane[:, None] * $(64)) + r154_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r154_value = tl.dot(r154_left, r154_right)
      r154_value = r154_value.to(tl.float32)
      tl.store($(candidate_vbmm_45_ptr) + ((r154_group * $(2048)) + (r154_row[:, None] * $(64)) + r154_col[None, :]), r154_value)
    }
    r155_z = tl.arange($(16384))
    r155_value = (tl.load($(candidate_vbmm_45_ptr) + ((((r155_z / $(2048)) % $(8)) * $(2048)) + (((r155_z / $(64)) % $(32)) * $(64)) + (r155_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_3_ptr) + ((((r155_z / $(2048)) % $(8)) * $(32)) + ((r155_z / $(64)) % $(32))), dtype=tl.float32))
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(16384)])
    tl.store($(candidate_vsub_31_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(16384))
    r156_value = (tl.load($(candidate_vexp_1_ptr) + ((((r156_z / $(2048)) % $(8)) * $(2048)) + (((r156_z / $(64)) % $(32)) * $(64)) + (r156_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_31_ptr) + ((((r156_z / $(2048)) % $(8)) * $(2048)) + (((r156_z / $(64)) % $(32)) * $(64)) + (r156_z % $(64))), dtype=tl.float32))
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(16384)])
    tl.store($(candidate_vmul_39_ptr) + r156_z, r156_value)
    r157_row = tl.arange($(32))
    r157_col = tl.arange($(192))
    r157_lane = tl.arange($(64))
    for r157_group in range($(8)) {
      r157_left = tl.load($(candidate_vmul_39_ptr) + (((((((r157_group % $(8)) * $(2048)) + (r157_row[:, None] * $(64)) + r157_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r157_group % $(8)) * $(2048)) + (r157_row[:, None] * $(64)) + r157_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r157_group % $(8)) * $(2048)) + (r157_row[:, None] * $(64)) + r157_lane[None, :]) % $(64))), dtype=tl.float32)
      r157_right = tl.load($(candidate_vto_copy_82_ptr) + (((((((r157_group % $(8)) * $(12288)) + (r157_lane[:, None] * $(192)) + r157_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r157_group % $(8)) * $(12288)) + (r157_lane[:, None] * $(192)) + r157_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r157_group % $(8)) * $(12288)) + (r157_lane[:, None] * $(192)) + r157_col[None, :]) % $(192))), dtype=tl.float32)
      r157_value = tl.dot(r157_left, r157_right)
      r157_value = r157_value.to(tl.float32)
      tl.store($(candidate_vbmm_46_ptr) + ((r157_group * $(6144)) + (r157_row[:, None] * $(192)) + r157_col[None, :]), r157_value)
    }
    r158_z = tl.arange($(49152))
    r158_value = (tl.load($(candidate_vbmm_46_ptr) + ((((r158_z / $(6144)) % $(8)) * $(6144)) + (((r158_z / $(192)) % $(32)) * $(192)) + (r158_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(49152)])
    tl.store($(candidate_vmul_40_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(49152))
    r159_value = tl.load($(candidate_vmul_40_ptr) + ((((r159_z / $(192)) % $(8)) * $(6144)) + (((r159_z / $(1536)) % $(32)) * $(192)) + (r159_z % $(192))), dtype=tl.float32)
    r159_value = r159_value.to(tl.bfloat16)
    r159_value = tl.broadcast_to(r159_value, [$(49152)])
    tl.store($(candidate_vto_copy_86_ptr) + r159_z, r159_value)
    r160_row = tl.arange($(64))
    r160_col = tl.arange($(192))
    r160_lane = tl.arange($(32))
    for r160_group in range($(8)) {
      r160_left = tl.load($(candidate_vmul_39_ptr) + (((((((r160_group % $(8)) * $(2048)) + (r160_row[:, None] * $(32)) + r160_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r160_group % $(8)) * $(2048)) + (r160_row[:, None] * $(32)) + r160_lane[None, :]) % $(32)) * $(64)) + (((((r160_group % $(8)) * $(2048)) + (r160_row[:, None] * $(32)) + r160_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r160_right = tl.load($(candidate_vto_copy_81_ptr) + (((((((r160_group % $(8)) * $(6144)) + (r160_lane[:, None] * $(192)) + r160_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r160_group % $(8)) * $(6144)) + (r160_lane[:, None] * $(192)) + r160_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r160_group % $(8)) * $(6144)) + (r160_lane[:, None] * $(192)) + r160_col[None, :]) % $(192))), dtype=tl.float32)
      r160_value = tl.dot(r160_left, r160_right)
      r160_value = r160_value.to(tl.float32)
      tl.store($(candidate_vbmm_47_ptr) + ((r160_group * $(12288)) + (r160_row[:, None] * $(192)) + r160_col[None, :]), r160_value)
    }
    r161_z = tl.arange($(98304))
    r161_value = (tl.load($(candidate_vbmm_47_ptr) + ((((r161_z / $(12288)) % $(8)) * $(12288)) + (((r161_z / $(192)) % $(64)) * $(192)) + (r161_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(98304)])
    tl.store($(candidate_vmul_41_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(98304))
    r162_value = tl.load($(candidate_vmul_41_ptr) + ((((r162_z / $(192)) % $(8)) * $(12288)) + (((r162_z / $(1536)) % $(64)) * $(192)) + (r162_z % $(192))), dtype=tl.float32)
    r162_value = r162_value.to(tl.bfloat16)
    r162_value = tl.broadcast_to(r162_value, [$(98304)])
    tl.store($(candidate_vto_copy_87_ptr) + r162_z, r162_value)
    r163_row = tl.arange($(64))
    r163_col = tl.arange($(128))
    r163_lane = tl.arange($(32))
    for r163_group in range($(8)) {
      r163_left = tl.load($(candidate_vexp_1_ptr) + (((((((r163_group % $(8)) * $(2048)) + (r163_row[:, None] * $(32)) + r163_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r163_group % $(8)) * $(2048)) + (r163_row[:, None] * $(32)) + r163_lane[None, :]) % $(32)) * $(64)) + (((((r163_group % $(8)) * $(2048)) + (r163_row[:, None] * $(32)) + r163_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r163_right = tl.load($(candidate_vto_copy_84_ptr) + (((((((r163_group % $(8)) * $(4096)) + (r163_lane[:, None] * $(128)) + r163_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r163_group % $(8)) * $(4096)) + (r163_lane[:, None] * $(128)) + r163_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r163_group % $(8)) * $(4096)) + (r163_lane[:, None] * $(128)) + r163_col[None, :]) % $(128))), dtype=tl.float32)
      r163_value = tl.dot(r163_left, r163_right)
      r163_value = r163_value.to(tl.float32)
      tl.store($(candidate_vbmm_48_ptr) + ((r163_group * $(8192)) + (r163_row[:, None] * $(128)) + r163_col[None, :]), r163_value)
    }
    r164_z = tl.arange($(65536))
    r164_value = tl.load($(candidate_vbmm_48_ptr) + ((((r164_z / $(128)) % $(8)) * $(8192)) + (((r164_z / $(1024)) % $(64)) * $(128)) + (r164_z % $(128))), dtype=tl.float32)
    r164_value = r164_value.to(tl.bfloat16)
    r164_value = tl.broadcast_to(r164_value, [$(65536)])
    tl.store($(candidate_vto_copy_88_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(147456))
    r165_value = tl.where(((r165_z / $(1536)) % $(96)) < $(64), tl.load($(candidate_vto_copy_87_ptr) + ((((((r165_z / $(1536)) % $(96)) - $(0)) % $(64)) * $(1536)) + (((r165_z / $(192)) % $(8)) * $(192)) + (r165_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r165_value = r165_value.to(tl.bfloat16)
    r165_value = tl.broadcast_to(r165_value, [$(147456)])
    tl.store($(candidate_vcat_30_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(98304))
    r166_value = tl.where(((r166_z / $(1024)) % $(96)) < $(64), tl.load($(candidate_vto_copy_88_ptr) + ((((((r166_z / $(1024)) % $(96)) - $(0)) % $(64)) * $(1024)) + (((r166_z / $(128)) % $(8)) * $(128)) + (r166_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r166_value = r166_value.to(tl.bfloat16)
    r166_value = tl.broadcast_to(r166_value, [$(98304)])
    tl.store($(candidate_vcat_31_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(49152))
    r167_value = tl.load($(candidate_vto_copy_86_ptr) + ((((((r167_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(1536)) + (((r167_z / $(192)) % $(8)) * $(192)) + (r167_z % $(192))), dtype=tl.bfloat16)
    r167_value = r167_value.to(tl.bfloat16)
    r167_value = tl.broadcast_to(r167_value, [$(49152)])
    tl.store($(candidate_vcat_32_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(147456))
    r168_value = tl.load($(candidate_vcat_30_ptr) + ((((((r168_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r168_z / $(192)) % $(8)) * $(192)) + (r168_z % $(192))), dtype=tl.bfloat16)
    r168_value = r168_value.to(tl.bfloat16)
    r168_value = tl.broadcast_to(r168_value, [$(147456)])
    tl.store($(candidate_vcat_33_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(98304))
    r169_value = tl.load($(candidate_vcat_31_ptr) + ((((((r169_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r169_z / $(128)) % $(8)) * $(128)) + (r169_z % $(128))), dtype=tl.bfloat16)
    r169_value = r169_value.to(tl.bfloat16)
    r169_value = tl.broadcast_to(r169_value, [$(98304)])
    tl.store($(candidate_vcat_34_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(6144))
    r170_lane = tl.arange($(8))
    r170_value = tl.sum(tl.load($(candidate_vcat_33_ptr) + ((((r170_z[:, None] / $(64)) % $(96)) * $(1536)) + (r170_lane[None, :] * $(192)) + (r170_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r170_value = r170_value.to(tl.bfloat16)
    r170_value = tl.broadcast_to(r170_value, [$(6144)])
    tl.store($(candidate_vsum_4_ptr) + r170_z, r170_value)
    r171_row = tl.arange($(96))
    r171_col = tl.arange($(512))
    r171_lane = tl.arange($(1024))
    for r171_group in range($(1)) {
      r171_left = tl.load($(candidate_vcat_33_ptr) + ((((((r171_row[:, None] * $(1024)) + r171_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + ((((r171_row[:, None] * $(1024)) + r171_lane[None, :]) % $(8)) * $(192)) + ((((r171_row[:, None] * $(1024)) + r171_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r171_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r171_lane[:, None] * $(512)) + r171_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r171_lane[:, None] * $(512)) + r171_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r171_lane[:, None] * $(512)) + r171_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r171_value = tl.dot(r171_left, r171_right)
      r171_value = r171_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_49_ptr) + ((r171_group * $(49152)) + (r171_row[:, None] * $(512)) + r171_col[None, :]), r171_value)
    }
    r172_row = tl.arange($(96))
    r172_col = tl.arange($(512))
    r172_lane = tl.arange($(1024))
    for r172_group in range($(1)) {
      r172_left = tl.load($(candidate_vcat_34_ptr) + ((((((r172_row[:, None] * $(1024)) + r172_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + ((((r172_row[:, None] * $(1024)) + r172_lane[None, :]) % $(8)) * $(128)) + ((((r172_row[:, None] * $(1024)) + r172_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r172_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r172_lane[:, None] * $(512)) + r172_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r172_lane[:, None] * $(512)) + r172_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r172_lane[:, None] * $(512)) + r172_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r172_value = tl.dot(r172_left, r172_right)
      r172_value = r172_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_50_ptr) + ((r172_group * $(49152)) + (r172_row[:, None] * $(512)) + r172_col[None, :]), r172_value)
    }
    r173_z = tl.arange($(49152))
    r173_value = (tl.load($(candidate_vbmm_49_ptr) + ((((r173_z / $(512)) % $(96)) * $(512)) + (r173_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_50_ptr) + ((((r173_z / $(512)) % $(96)) * $(512)) + (r173_z % $(512))), dtype=tl.bfloat16))
    r173_value = r173_value.to(tl.bfloat16)
    r173_value = tl.broadcast_to(r173_value, [$(49152)])
    tl.store($(candidate_vadd_29_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(147456))
    r174_value = tl.where(((((r174_z / $(512)) % $(288)) >= $(192)) & (((r174_z / $(512)) % $(288)) < $(288)) & ((((r174_z / $(512)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vadd_29_ptr) + ((((((r174_z / $(512)) % $(288)) - $(192)) % $(96)) * $(512)) + (r174_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r174_value = r174_value.to(tl.bfloat16)
    r174_value = tl.broadcast_to(r174_value, [$(147456)])
    tl.store($(candidate_vslice_backward_4_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(147456))
    r175_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r175_z / $(512)) % $(288)) * $(512)) + (r175_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_4_ptr) + ((((r175_z / $(512)) % $(288)) * $(512)) + (r175_z % $(512))), dtype=tl.bfloat16))
    r175_value = r175_value.to(tl.bfloat16)
    r175_value = tl.broadcast_to(r175_value, [$(147456)])
    tl.store($(candidate_vadd_30_ptr) + r175_z, r175_value)
    r176_z = tl.arange($(18432))
    r176_value = tl.where(((((r176_z / $(64)) % $(288)) >= $(192)) & (((r176_z / $(64)) % $(288)) < $(288)) & ((((r176_z / $(64)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vsum_4_ptr) + ((((((r176_z / $(64)) % $(288)) - $(192)) % $(96)) * $(64)) + (r176_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r176_value = r176_value.to(tl.bfloat16)
    r176_value = tl.broadcast_to(r176_value, [$(18432)])
    tl.store($(candidate_vslice_backward_5_ptr) + r176_z, r176_value)
    r177_z = tl.arange($(18432))
    r177_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r177_z / $(64)) % $(288)) * $(64)) + (r177_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r177_z / $(64)) % $(288)) * $(64)) + (r177_z % $(64))), dtype=tl.bfloat16))
    r177_value = r177_value.to(tl.bfloat16)
    r177_value = tl.broadcast_to(r177_value, [$(18432)])
    tl.store($(candidate_vadd_31_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(294912))
    r178_value = tl.where(((((r178_z / $(1024)) % $(288)) >= $(224)) & (((r178_z / $(1024)) % $(288)) < $(256)) & ((((r178_z / $(1024)) % $(288)) - $(224)) % $(1) == $(0))), tl.load($(candidate_vcat_32_ptr) + ((((((r178_z / $(1024)) % $(288)) - $(224)) % $(32)) * $(1536)) + (((r178_z / $(128)) % $(8)) * $(192)) + (r178_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r178_value = r178_value.to(tl.bfloat16)
    r178_value = tl.broadcast_to(r178_value, [$(294912)])
    tl.store($(candidate_vslice_backward_6_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(294912))
    r179_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r179_z / $(1024)) % $(288)) * $(1024)) + (((r179_z / $(128)) % $(8)) * $(128)) + (r179_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_6_ptr) + ((((r179_z / $(1024)) % $(288)) * $(1024)) + (((r179_z / $(128)) % $(8)) * $(128)) + (r179_z % $(128))), dtype=tl.bfloat16))
    r179_value = r179_value.to(tl.bfloat16)
    r179_value = tl.broadcast_to(r179_value, [$(294912)])
    tl.store($(candidate_vadd_32_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(147456))
    r180_value = tl.where(((((r180_z / $(512)) % $(288)) >= $(224)) & (((r180_z / $(512)) % $(288)) < $(256)) & ((((r180_z / $(512)) % $(288)) - $(224)) % $(1) == $(0))), tl.load($(candidate_vcat_32_ptr) + ((((((r180_z / $(512)) % $(288)) - $(224)) % $(32)) * $(1536)) + (((r180_z / $(64)) % $(8)) * $(192)) + (r180_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r180_value = r180_value.to(tl.bfloat16)
    r180_value = tl.broadcast_to(r180_value, [$(147456)])
    tl.store($(candidate_vslice_backward_7_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(147456))
    r181_value = (tl.load($(candidate_vslice_backward_3_ptr) + ((((r181_z / $(512)) % $(288)) * $(512)) + (((r181_z / $(64)) % $(8)) * $(64)) + (r181_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_7_ptr) + ((((r181_z / $(512)) % $(288)) * $(512)) + (((r181_z / $(64)) % $(8)) * $(64)) + (r181_z % $(64))), dtype=tl.bfloat16))
    r181_value = r181_value.to(tl.bfloat16)
    r181_value = tl.broadcast_to(r181_value, [$(147456)])
    tl.store($(candidate_vadd_33_ptr) + r181_z, r181_value)
    r182_row = tl.arange($(96))
    r182_col = tl.arange($(1024))
    r182_lane = tl.arange($(512))
    for r182_group in range($(1)) {
      r182_left = tl.load($(candidate_vcat_7_ptr) + ((((((r182_row[:, None] * $(512)) + r182_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r182_row[:, None] * $(512)) + r182_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r182_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r182_lane[:, None] * $(1024)) + r182_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r182_lane[:, None] * $(1024)) + r182_col[None, :]) % $(128)) * $(512)) + ((((r182_lane[:, None] * $(1024)) + r182_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r182_value = tl.dot(r182_left, r182_right)
      r182_value = r182_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_53_ptr) + ((r182_group * $(98304)) + (r182_row[:, None] * $(1024)) + r182_col[None, :]), r182_value)
    }
    r183_row = tl.arange($(96))
    r183_col = tl.arange($(1024))
    r183_lane = tl.arange($(512))
    for r183_group in range($(1)) {
      r183_left = tl.load($(candidate_vcat_7_ptr) + ((((((r183_row[:, None] * $(512)) + r183_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r183_row[:, None] * $(512)) + r183_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r183_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r183_lane[:, None] * $(1024)) + r183_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r183_lane[:, None] * $(1024)) + r183_col[None, :]) % $(128)) * $(512)) + ((((r183_lane[:, None] * $(1024)) + r183_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r183_value = tl.dot(r183_left, r183_right)
      r183_value = r183_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_54_ptr) + ((r183_group * $(98304)) + (r183_row[:, None] * $(1024)) + r183_col[None, :]), r183_value)
    }
    r184_z = tl.arange($(49152))
    r184_value = tl.where((r184_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r184_z / $(1536)) % $(32)) * $(1024)) + (((r184_z / $(192)) % $(8)) * $(128)) + ((r184_z % $(192)) - $(0)) + $(196608)) / $(1024)) % $(288)) * $(1536)) + (((((((r184_z / $(1536)) % $(32)) * $(1024)) + (((r184_z / $(192)) % $(8)) * $(128)) + ((r184_z % $(192)) - $(0)) + $(196608)) / $(128)) % $(8)) * $(192)) + (((((r184_z / $(1536)) % $(32)) * $(1024)) + (((r184_z / $(192)) % $(8)) * $(128)) + ((r184_z % $(192)) - $(0)) + $(196608)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r184_z / $(1536)) % $(32)) * $(512)) + (((r184_z / $(192)) % $(8)) * $(64)) + ((r184_z % $(192)) - $(128)) + $(98304)) / $(64)) % $(8)) * $(18432)) + (((((((r184_z / $(1536)) % $(32)) * $(512)) + (((r184_z / $(192)) % $(8)) * $(64)) + ((r184_z % $(192)) - $(128)) + $(98304)) / $(512)) % $(288)) * $(64)) + (((((r184_z / $(1536)) % $(32)) * $(512)) + (((r184_z / $(192)) % $(8)) * $(64)) + ((r184_z % $(192)) - $(128)) + $(98304)) % $(64))), dtype=tl.bfloat16))
    r184_value = r184_value.to(tl.bfloat16)
    r184_value = tl.broadcast_to(r184_value, [$(49152)])
    tl.store($(candidate_vcat_36_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(147456))
    r185_value = tl.where((r185_z % $(192)) < $(128), tl.load($(candidate_vbmm_53_ptr) + ((((((((r185_z / $(1536)) % $(96)) * $(1024)) + (((r185_z / $(192)) % $(8)) * $(128)) + ((r185_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r185_z / $(1536)) % $(96)) * $(1024)) + (((r185_z / $(192)) % $(8)) * $(128)) + ((r185_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r185_z / $(1536)) % $(96)) * $(1024)) + (((r185_z / $(192)) % $(8)) * $(128)) + ((r185_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r185_z / $(1536)) % $(96)) * $(576)) + ((((r185_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r185_value = r185_value.to(tl.bfloat16)
    r185_value = tl.broadcast_to(r185_value, [$(147456)])
    tl.store($(candidate_vcat_37_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(49152))
    r186_value = tl.load($(candidate_vcat_36_ptr) + ((((r186_z / $(192)) % $(32)) * $(1536)) + (((r186_z / $(6144)) % $(8)) * $(192)) + (r186_z % $(192))), dtype=tl.bfloat16)
    r186_value = r186_value.to(tl.float32)
    r186_value = tl.broadcast_to(r186_value, [$(49152)])
    tl.store($(candidate_vto_copy_89_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(49152))
    r187_value = tl.load($(candidate_vcat_37_ptr) + ((((r187_z / $(192)) % $(32)) * $(1536)) + (((r187_z / $(6144)) % $(8)) * $(192)) + (r187_z % $(192))), dtype=tl.bfloat16)
    r187_value = r187_value.to(tl.float32)
    r187_value = tl.broadcast_to(r187_value, [$(49152)])
    tl.store($(candidate_vto_copy_90_ptr) + r187_z, r187_value)
    r188_row = tl.arange($(32))
    r188_col = tl.arange($(32))
    r188_lane = tl.arange($(192))
    for r188_group in range($(8)) {
      r188_left = tl.load($(candidate_vto_copy_89_ptr) + (((((((r188_group % $(8)) * $(6144)) + (r188_row[:, None] * $(192)) + r188_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r188_group % $(8)) * $(6144)) + (r188_row[:, None] * $(192)) + r188_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r188_group % $(8)) * $(6144)) + (r188_row[:, None] * $(192)) + r188_lane[None, :]) % $(192))), dtype=tl.float32)
      r188_right = tl.load($(candidate_vto_copy_90_ptr) + (((((((r188_group % $(8)) * $(6144)) + (r188_lane[:, None] * $(32)) + r188_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r188_group % $(8)) * $(6144)) + (r188_lane[:, None] * $(32)) + r188_col[None, :]) % $(32)) * $(192)) + (((((r188_group % $(8)) * $(6144)) + (r188_lane[:, None] * $(32)) + r188_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r188_value = tl.dot(r188_left, r188_right)
      r188_value = r188_value.to(tl.float32)
      tl.store($(candidate_vbmm_55_ptr) + ((r188_group * $(1024)) + (r188_row[:, None] * $(32)) + r188_col[None, :]), r188_value)
    }
    r189_z = tl.arange($(8192))
    r189_value = (tl.load($(candidate_vbmm_55_ptr) + ((((r189_z / $(1024)) % $(8)) * $(1024)) + (((r189_z / $(32)) % $(32)) * $(32)) + (r189_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r189_value = r189_value.to(tl.float32)
    r189_value = tl.broadcast_to(r189_value, [$(8192)])
    tl.store($(candidate_vmul_42_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(32))
    r190_value = (tl.cast(r190_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r190_value = tl.broadcast_to(r190_value, [$(32)])
    tl.store($(candidate_vadd_34_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(32))
    r191_value = (tl.load($(candidate_vadd_34_ptr) + r191_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r191_value = tl.broadcast_to(r191_value, [$(32)])
    tl.store($(candidate_vsub_32_ptr) + r191_z, r191_value)
    r192_z = tl.arange($(1024))
    r192_value = (tl.cast((r192_z % $(32)), tl.int64) > tl.load($(candidate_vsub_32_ptr) + ((r192_z / $(32)) % $(32)), dtype=tl.int64))
    r192_value = tl.broadcast_to(r192_value, [$(1024)])
    tl.store($(candidate_vgt_13_ptr) + r192_z, r192_value)
    r193_z = tl.arange($(8192))
    r193_value = tl.where(tl.load($(candidate_vgt_13_ptr) + ((((r193_z / $(32)) % $(32)) * $(32)) + (r193_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_42_ptr) + ((((r193_z / $(1024)) % $(8)) * $(1024)) + (((r193_z / $(32)) % $(32)) * $(32)) + (r193_z % $(32))), dtype=tl.float32))
    r193_value = r193_value.to(tl.float32)
    r193_value = tl.broadcast_to(r193_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_13_ptr) + r193_z, r193_value)
    r194_z = tl.arange($(8192))
    r194_lane = tl.arange($(32))
    r194_row_values = tl.load($(candidate_vmasked_fill_13_ptr) + ((((r194_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r194_z[:, None] / $(32)) % $(32)) * $(32)) + (r194_lane[None, :] % $(32))), dtype=tl.float32)
    r194_maximum = tl.max(r194_row_values, axis=1, keep_dims=true)
    r194_weights = tl.exp(r194_row_values - r194_maximum)
    r194_total = tl.sum(r194_weights, axis=1)
    r194_maximum = tl.max(r194_row_values, axis=1)
    r194_value = tl.exp(tl.load($(candidate_vmasked_fill_13_ptr) + ((((r194_z / $(1024)) % $(8)) * $(1024)) + (((r194_z / $(32)) % $(32)) * $(32)) + (r194_z % $(32))), dtype=tl.float32) - r194_maximum) / r194_total
    r194_value = r194_value.to(tl.float32)
    r194_value = tl.broadcast_to(r194_value, [$(8192)])
    tl.store($(candidate_vsoftmax_11_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(8192))
    r195_value = tl.load($(candidate_vsoftmax_11_ptr) + ((((r195_z / $(1024)) % $(8)) * $(1024)) + (((r195_z / $(32)) % $(32)) * $(32)) + (r195_z % $(32))), dtype=tl.float32)
    r195_value = r195_value.to(tl.bfloat16)
    r195_value = tl.broadcast_to(r195_value, [$(8192)])
    tl.store($(candidate_vto_copy_91_ptr) + r195_z, r195_value)
    r196_row = tl.arange($(32))
    r196_col = tl.arange($(128))
    r196_lane = tl.arange($(32))
    for r196_group in range($(8)) {
      r196_left = tl.load($(candidate_vto_copy_91_ptr) + (((((((r196_group % $(8)) * $(1024)) + (r196_row[:, None] * $(32)) + r196_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r196_group % $(8)) * $(1024)) + (r196_row[:, None] * $(32)) + r196_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r196_group % $(8)) * $(1024)) + (r196_row[:, None] * $(32)) + r196_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r196_right = tl.load($(candidate_vbmm_54_ptr) + (((((((r196_group % $(8)) * $(4096)) + (r196_lane[:, None] * $(128)) + r196_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r196_group % $(8)) * $(4096)) + (r196_lane[:, None] * $(128)) + r196_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r196_group % $(8)) * $(4096)) + (r196_lane[:, None] * $(128)) + r196_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r196_value = tl.dot(r196_left, r196_right)
      r196_value = r196_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_56_ptr) + ((r196_group * $(4096)) + (r196_row[:, None] * $(128)) + r196_col[None, :]), r196_value)
    }
    r197_z = tl.arange($(256))
    r197_lane = tl.arange($(32))
    r197_row_values = tl.load($(candidate_vmasked_fill_13_ptr) + ((((r197_z[:, None] / $(32)) % $(8)) * $(1024)) + ((r197_z[:, None] % $(32)) * $(32)) + (r197_lane[None, :] % $(32))), dtype=tl.float32)
    r197_maximum = tl.max(r197_row_values, axis=1, keep_dims=true)
    r197_weights = tl.exp(r197_row_values - r197_maximum)
    r197_total = tl.sum(r197_weights, axis=1)
    r197_maximum = tl.max(r197_row_values, axis=1)
    r197_value = r197_maximum + tl.log(r197_total)
    r197_value = r197_value.to(tl.float32)
    r197_value = tl.broadcast_to(r197_value, [$(256)])
    tl.store($(candidate_vlogsumexp_11_ptr) + r197_z, r197_value)
    r198_z = tl.arange($(32768))
    r198_value = tl.load($(candidate_vbmm_56_ptr) + ((((((((r198_z / $(128)) % $(8)) * $(4096)) + ((((r198_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r198_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r198_z / $(128)) % $(8)) * $(32)) + (((r198_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r198_z % $(128))), dtype=tl.bfloat16)
    r198_value = r198_value.to(tl.bfloat16)
    r198_value = tl.broadcast_to(r198_value, [$(32768)])
    tl.store($(candidate_vcat_38_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(256))
    r199_value = tl.load($(candidate_vlogsumexp_11_ptr) + (((r199_z % $(8)) * $(32)) + ((((r199_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r199_value = r199_value.to(tl.float32)
    r199_value = tl.broadcast_to(r199_value, [$(256)])
    tl.store($(candidate_vcat_39_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(49152))
    r200_value = tl.load($(candidate_vcat_36_ptr) + ((((r200_z / $(192)) % $(32)) * $(1536)) + (((r200_z / $(6144)) % $(8)) * $(192)) + (r200_z % $(192))), dtype=tl.bfloat16)
    r200_value = r200_value.to(tl.float32)
    r200_value = tl.broadcast_to(r200_value, [$(49152)])
    tl.store($(candidate_vto_copy_92_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(49152))
    r201_value = tl.load($(candidate_vcat_37_ptr) + ((((r201_z / $(192)) % $(32)) * $(1536)) + (((r201_z / $(6144)) % $(8)) * $(192)) + (r201_z % $(192))), dtype=tl.bfloat16)
    r201_value = r201_value.to(tl.float32)
    r201_value = tl.broadcast_to(r201_value, [$(49152)])
    tl.store($(candidate_vto_copy_93_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(32768))
    r202_value = tl.load($(candidate_vbmm_54_ptr) + ((((r202_z / $(128)) % $(32)) * $(1024)) + (((r202_z / $(4096)) % $(8)) * $(128)) + (r202_z % $(128))), dtype=tl.bfloat16)
    r202_value = r202_value.to(tl.float32)
    r202_value = tl.broadcast_to(r202_value, [$(32768)])
    tl.store($(candidate_vto_copy_94_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(32768))
    r203_value = tl.load($(candidate_vmm_22_ptr) + ((((r203_z / $(128)) % $(32)) * $(1024)) + (((r203_z / $(4096)) % $(8)) * $(128)) + (r203_z % $(128)) + $(196608)), dtype=tl.bfloat16)
    r203_value = r203_value.to(tl.float32)
    r203_value = tl.broadcast_to(r203_value, [$(32768)])
    tl.store($(candidate_vto_copy_95_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(32768))
    r204_value = tl.load($(candidate_vcat_38_ptr) + ((((r204_z / $(128)) % $(32)) * $(1024)) + (((r204_z / $(4096)) % $(8)) * $(128)) + (r204_z % $(128))), dtype=tl.bfloat16)
    r204_value = r204_value.to(tl.float32)
    r204_value = tl.broadcast_to(r204_value, [$(32768)])
    tl.store($(candidate_vto_copy_96_ptr) + r204_z, r204_value)
    r205_row = tl.arange($(32))
    r205_col = tl.arange($(32))
    r205_lane = tl.arange($(192))
    for r205_group in range($(8)) {
      r205_left = tl.load($(candidate_vto_copy_92_ptr) + (((((((r205_group % $(8)) * $(6144)) + (r205_row[:, None] * $(192)) + r205_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r205_group % $(8)) * $(6144)) + (r205_row[:, None] * $(192)) + r205_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r205_group % $(8)) * $(6144)) + (r205_row[:, None] * $(192)) + r205_lane[None, :]) % $(192))), dtype=tl.float32)
      r205_right = tl.load($(candidate_vto_copy_93_ptr) + (((((((r205_group % $(8)) * $(6144)) + (r205_lane[:, None] * $(32)) + r205_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r205_group % $(8)) * $(6144)) + (r205_lane[:, None] * $(32)) + r205_col[None, :]) % $(32)) * $(192)) + (((((r205_group % $(8)) * $(6144)) + (r205_lane[:, None] * $(32)) + r205_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r205_value = tl.dot(r205_left, r205_right)
      r205_value = r205_value.to(tl.float32)
      tl.store($(candidate_vbmm_57_ptr) + ((r205_group * $(1024)) + (r205_row[:, None] * $(32)) + r205_col[None, :]), r205_value)
    }
    r206_z = tl.arange($(8192))
    r206_value = (tl.load($(candidate_vbmm_57_ptr) + ((((r206_z / $(1024)) % $(8)) * $(1024)) + (((r206_z / $(32)) % $(32)) * $(32)) + (r206_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r206_value = r206_value.to(tl.float32)
    r206_value = tl.broadcast_to(r206_value, [$(8192)])
    tl.store($(candidate_vmul_43_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(32))
    r207_value = (tl.cast(r207_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r207_value = tl.broadcast_to(r207_value, [$(32)])
    tl.store($(candidate_vadd_35_ptr) + r207_z, r207_value)
    r208_z = tl.arange($(32))
    r208_value = (tl.load($(candidate_vadd_35_ptr) + r208_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r208_value = tl.broadcast_to(r208_value, [$(32)])
    tl.store($(candidate_vsub_33_ptr) + r208_z, r208_value)
    r209_z = tl.arange($(1024))
    r209_value = (tl.cast((r209_z % $(32)), tl.int64) > tl.load($(candidate_vsub_33_ptr) + ((r209_z / $(32)) % $(32)), dtype=tl.int64))
    r209_value = tl.broadcast_to(r209_value, [$(1024)])
    tl.store($(candidate_vgt_14_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(8192))
    r210_value = tl.where(tl.load($(candidate_vgt_14_ptr) + ((((r210_z / $(32)) % $(32)) * $(32)) + (r210_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_43_ptr) + ((((r210_z / $(1024)) % $(8)) * $(1024)) + (((r210_z / $(32)) % $(32)) * $(32)) + (r210_z % $(32))), dtype=tl.float32))
    r210_value = r210_value.to(tl.float32)
    r210_value = tl.broadcast_to(r210_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_14_ptr) + r210_z, r210_value)
    r211_z = tl.arange($(8192))
    r211_value = (tl.load($(candidate_vmasked_fill_14_ptr) + ((((r211_z / $(1024)) % $(8)) * $(1024)) + (((r211_z / $(32)) % $(32)) * $(32)) + (r211_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_39_ptr) + ((((r211_z / $(32)) % $(32)) * $(8)) + ((r211_z / $(1024)) % $(8))), dtype=tl.float32))
    r211_value = r211_value.to(tl.float32)
    r211_value = tl.broadcast_to(r211_value, [$(8192)])
    tl.store($(candidate_vsub_34_ptr) + r211_z, r211_value)
    r212_z = tl.arange($(8192))
    r212_value = tl.exp(tl.load($(candidate_vsub_34_ptr) + ((((r212_z / $(1024)) % $(8)) * $(1024)) + (((r212_z / $(32)) % $(32)) * $(32)) + (r212_z % $(32))), dtype=tl.float32))
    r212_value = r212_value.to(tl.float32)
    r212_value = tl.broadcast_to(r212_value, [$(8192)])
    tl.store($(candidate_vexp_2_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(32768))
    r213_value = (tl.load($(candidate_vto_copy_95_ptr) + ((((r213_z / $(4096)) % $(8)) * $(4096)) + (((r213_z / $(128)) % $(32)) * $(128)) + (r213_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_96_ptr) + ((((r213_z / $(4096)) % $(8)) * $(4096)) + (((r213_z / $(128)) % $(32)) * $(128)) + (r213_z % $(128))), dtype=tl.float32))
    r213_value = r213_value.to(tl.float32)
    r213_value = tl.broadcast_to(r213_value, [$(32768)])
    tl.store($(candidate_vmul_44_ptr) + r213_z, r213_value)
    r214_z = tl.arange($(256))
    r214_lane = tl.arange($(128))
    r214_value = tl.sum(tl.load($(candidate_vmul_44_ptr) + ((((r214_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r214_z[:, None] % $(32)) * $(128)) + r214_lane[None, :]), dtype=tl.float32), axis=1)
    r214_value = r214_value.to(tl.float32)
    r214_value = tl.broadcast_to(r214_value, [$(256)])
    tl.store($(candidate_vsum_5_ptr) + r214_z, r214_value)
    r215_row = tl.arange($(32))
    r215_col = tl.arange($(32))
    r215_lane = tl.arange($(128))
    for r215_group in range($(8)) {
      r215_left = tl.load($(candidate_vto_copy_95_ptr) + (((((((r215_group % $(8)) * $(4096)) + (r215_row[:, None] * $(128)) + r215_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r215_group % $(8)) * $(4096)) + (r215_row[:, None] * $(128)) + r215_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r215_group % $(8)) * $(4096)) + (r215_row[:, None] * $(128)) + r215_lane[None, :]) % $(128))), dtype=tl.float32)
      r215_right = tl.load($(candidate_vto_copy_94_ptr) + (((((((r215_group % $(8)) * $(4096)) + (r215_lane[:, None] * $(32)) + r215_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r215_group % $(8)) * $(4096)) + (r215_lane[:, None] * $(32)) + r215_col[None, :]) % $(32)) * $(128)) + (((((r215_group % $(8)) * $(4096)) + (r215_lane[:, None] * $(32)) + r215_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r215_value = tl.dot(r215_left, r215_right)
      r215_value = r215_value.to(tl.float32)
      tl.store($(candidate_vbmm_58_ptr) + ((r215_group * $(1024)) + (r215_row[:, None] * $(32)) + r215_col[None, :]), r215_value)
    }
    r216_z = tl.arange($(8192))
    r216_value = (tl.load($(candidate_vbmm_58_ptr) + ((((r216_z / $(1024)) % $(8)) * $(1024)) + (((r216_z / $(32)) % $(32)) * $(32)) + (r216_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_5_ptr) + ((((r216_z / $(1024)) % $(8)) * $(32)) + ((r216_z / $(32)) % $(32))), dtype=tl.float32))
    r216_value = r216_value.to(tl.float32)
    r216_value = tl.broadcast_to(r216_value, [$(8192)])
    tl.store($(candidate_vsub_35_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(8192))
    r217_value = (tl.load($(candidate_vexp_2_ptr) + ((((r217_z / $(1024)) % $(8)) * $(1024)) + (((r217_z / $(32)) % $(32)) * $(32)) + (r217_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_35_ptr) + ((((r217_z / $(1024)) % $(8)) * $(1024)) + (((r217_z / $(32)) % $(32)) * $(32)) + (r217_z % $(32))), dtype=tl.float32))
    r217_value = r217_value.to(tl.float32)
    r217_value = tl.broadcast_to(r217_value, [$(8192)])
    tl.store($(candidate_vmul_45_ptr) + r217_z, r217_value)
    r218_row = tl.arange($(32))
    r218_col = tl.arange($(192))
    r218_lane = tl.arange($(32))
    for r218_group in range($(8)) {
      r218_left = tl.load($(candidate_vmul_45_ptr) + (((((((r218_group % $(8)) * $(1024)) + (r218_row[:, None] * $(32)) + r218_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r218_group % $(8)) * $(1024)) + (r218_row[:, None] * $(32)) + r218_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r218_group % $(8)) * $(1024)) + (r218_row[:, None] * $(32)) + r218_lane[None, :]) % $(32))), dtype=tl.float32)
      r218_right = tl.load($(candidate_vto_copy_93_ptr) + (((((((r218_group % $(8)) * $(6144)) + (r218_lane[:, None] * $(192)) + r218_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r218_group % $(8)) * $(6144)) + (r218_lane[:, None] * $(192)) + r218_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r218_group % $(8)) * $(6144)) + (r218_lane[:, None] * $(192)) + r218_col[None, :]) % $(192))), dtype=tl.float32)
      r218_value = tl.dot(r218_left, r218_right)
      r218_value = r218_value.to(tl.float32)
      tl.store($(candidate_vbmm_59_ptr) + ((r218_group * $(6144)) + (r218_row[:, None] * $(192)) + r218_col[None, :]), r218_value)
    }
    r219_z = tl.arange($(49152))
    r219_value = (tl.load($(candidate_vbmm_59_ptr) + ((((r219_z / $(6144)) % $(8)) * $(6144)) + (((r219_z / $(192)) % $(32)) * $(192)) + (r219_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r219_value = r219_value.to(tl.float32)
    r219_value = tl.broadcast_to(r219_value, [$(49152)])
    tl.store($(candidate_vmul_46_ptr) + r219_z, r219_value)
    r220_z = tl.arange($(49152))
    r220_value = tl.load($(candidate_vmul_46_ptr) + ((((r220_z / $(192)) % $(8)) * $(6144)) + (((r220_z / $(1536)) % $(32)) * $(192)) + (r220_z % $(192))), dtype=tl.float32)
    r220_value = r220_value.to(tl.bfloat16)
    r220_value = tl.broadcast_to(r220_value, [$(49152)])
    tl.store($(candidate_vto_copy_97_ptr) + r220_z, r220_value)
    r221_row = tl.arange($(32))
    r221_col = tl.arange($(192))
    r221_lane = tl.arange($(32))
    for r221_group in range($(8)) {
      r221_left = tl.load($(candidate_vmul_45_ptr) + (((((((r221_group % $(8)) * $(1024)) + (r221_row[:, None] * $(32)) + r221_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r221_group % $(8)) * $(1024)) + (r221_row[:, None] * $(32)) + r221_lane[None, :]) % $(32)) * $(32)) + (((((r221_group % $(8)) * $(1024)) + (r221_row[:, None] * $(32)) + r221_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r221_right = tl.load($(candidate_vto_copy_92_ptr) + (((((((r221_group % $(8)) * $(6144)) + (r221_lane[:, None] * $(192)) + r221_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r221_group % $(8)) * $(6144)) + (r221_lane[:, None] * $(192)) + r221_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r221_group % $(8)) * $(6144)) + (r221_lane[:, None] * $(192)) + r221_col[None, :]) % $(192))), dtype=tl.float32)
      r221_value = tl.dot(r221_left, r221_right)
      r221_value = r221_value.to(tl.float32)
      tl.store($(candidate_vbmm_60_ptr) + ((r221_group * $(6144)) + (r221_row[:, None] * $(192)) + r221_col[None, :]), r221_value)
    }
    r222_z = tl.arange($(49152))
    r222_value = (tl.load($(candidate_vbmm_60_ptr) + ((((r222_z / $(6144)) % $(8)) * $(6144)) + (((r222_z / $(192)) % $(32)) * $(192)) + (r222_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r222_value = r222_value.to(tl.float32)
    r222_value = tl.broadcast_to(r222_value, [$(49152)])
    tl.store($(candidate_vmul_47_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(49152))
    r223_value = tl.load($(candidate_vmul_47_ptr) + ((((r223_z / $(192)) % $(8)) * $(6144)) + (((r223_z / $(1536)) % $(32)) * $(192)) + (r223_z % $(192))), dtype=tl.float32)
    r223_value = r223_value.to(tl.bfloat16)
    r223_value = tl.broadcast_to(r223_value, [$(49152)])
    tl.store($(candidate_vto_copy_98_ptr) + r223_z, r223_value)
    r224_row = tl.arange($(32))
    r224_col = tl.arange($(128))
    r224_lane = tl.arange($(32))
    for r224_group in range($(8)) {
      r224_left = tl.load($(candidate_vexp_2_ptr) + (((((((r224_group % $(8)) * $(1024)) + (r224_row[:, None] * $(32)) + r224_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r224_group % $(8)) * $(1024)) + (r224_row[:, None] * $(32)) + r224_lane[None, :]) % $(32)) * $(32)) + (((((r224_group % $(8)) * $(1024)) + (r224_row[:, None] * $(32)) + r224_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r224_right = tl.load($(candidate_vto_copy_95_ptr) + (((((((r224_group % $(8)) * $(4096)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r224_group % $(8)) * $(4096)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r224_group % $(8)) * $(4096)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) % $(128))), dtype=tl.float32)
      r224_value = tl.dot(r224_left, r224_right)
      r224_value = r224_value.to(tl.float32)
      tl.store($(candidate_vbmm_61_ptr) + ((r224_group * $(4096)) + (r224_row[:, None] * $(128)) + r224_col[None, :]), r224_value)
    }
    r225_z = tl.arange($(32768))
    r225_value = tl.load($(candidate_vbmm_61_ptr) + ((((r225_z / $(128)) % $(8)) * $(4096)) + (((r225_z / $(1024)) % $(32)) * $(128)) + (r225_z % $(128))), dtype=tl.float32)
    r225_value = r225_value.to(tl.bfloat16)
    r225_value = tl.broadcast_to(r225_value, [$(32768)])
    tl.store($(candidate_vto_copy_99_ptr) + r225_z, r225_value)
    r226_z = tl.arange($(147456))
    r226_value = tl.where(((r226_z / $(1536)) % $(96)) < $(32), tl.load($(candidate_vto_copy_98_ptr) + ((((((r226_z / $(1536)) % $(96)) - $(0)) % $(32)) * $(1536)) + (((r226_z / $(192)) % $(8)) * $(192)) + (r226_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r226_value = r226_value.to(tl.bfloat16)
    r226_value = tl.broadcast_to(r226_value, [$(147456)])
    tl.store($(candidate_vcat_40_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(98304))
    r227_value = tl.where(((r227_z / $(1024)) % $(96)) < $(32), tl.load($(candidate_vto_copy_99_ptr) + ((((((r227_z / $(1024)) % $(96)) - $(0)) % $(32)) * $(1024)) + (((r227_z / $(128)) % $(8)) * $(128)) + (r227_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r227_value = r227_value.to(tl.bfloat16)
    r227_value = tl.broadcast_to(r227_value, [$(98304)])
    tl.store($(candidate_vcat_41_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(49152))
    r228_value = tl.load($(candidate_vto_copy_97_ptr) + ((((((r228_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(1536)) + (((r228_z / $(192)) % $(8)) * $(192)) + (r228_z % $(192))), dtype=tl.bfloat16)
    r228_value = r228_value.to(tl.bfloat16)
    r228_value = tl.broadcast_to(r228_value, [$(49152)])
    tl.store($(candidate_vcat_42_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(147456))
    r229_value = tl.load($(candidate_vcat_40_ptr) + ((((((r229_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r229_z / $(192)) % $(8)) * $(192)) + (r229_z % $(192))), dtype=tl.bfloat16)
    r229_value = r229_value.to(tl.bfloat16)
    r229_value = tl.broadcast_to(r229_value, [$(147456)])
    tl.store($(candidate_vcat_43_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(98304))
    r230_value = tl.load($(candidate_vcat_41_ptr) + ((((((r230_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r230_z / $(128)) % $(8)) * $(128)) + (r230_z % $(128))), dtype=tl.bfloat16)
    r230_value = r230_value.to(tl.bfloat16)
    r230_value = tl.broadcast_to(r230_value, [$(98304)])
    tl.store($(candidate_vcat_44_ptr) + r230_z, r230_value)
    r231_z = tl.arange($(6144))
    r231_lane = tl.arange($(8))
    r231_value = tl.sum(tl.load($(candidate_vcat_43_ptr) + ((((r231_z[:, None] / $(64)) % $(96)) * $(1536)) + (r231_lane[None, :] * $(192)) + (r231_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r231_value = r231_value.to(tl.bfloat16)
    r231_value = tl.broadcast_to(r231_value, [$(6144)])
    tl.store($(candidate_vsum_6_ptr) + r231_z, r231_value)
    r232_row = tl.arange($(96))
    r232_col = tl.arange($(512))
    r232_lane = tl.arange($(1024))
    for r232_group in range($(1)) {
      r232_left = tl.load($(candidate_vcat_43_ptr) + ((((((r232_row[:, None] * $(1024)) + r232_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + ((((r232_row[:, None] * $(1024)) + r232_lane[None, :]) % $(8)) * $(192)) + ((((r232_row[:, None] * $(1024)) + r232_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r232_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r232_lane[:, None] * $(512)) + r232_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r232_lane[:, None] * $(512)) + r232_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r232_lane[:, None] * $(512)) + r232_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r232_value = tl.dot(r232_left, r232_right)
      r232_value = r232_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_62_ptr) + ((r232_group * $(49152)) + (r232_row[:, None] * $(512)) + r232_col[None, :]), r232_value)
    }
    r233_row = tl.arange($(96))
    r233_col = tl.arange($(512))
    r233_lane = tl.arange($(1024))
    for r233_group in range($(1)) {
      r233_left = tl.load($(candidate_vcat_44_ptr) + ((((((r233_row[:, None] * $(1024)) + r233_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + ((((r233_row[:, None] * $(1024)) + r233_lane[None, :]) % $(8)) * $(128)) + ((((r233_row[:, None] * $(1024)) + r233_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r233_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r233_lane[:, None] * $(512)) + r233_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r233_lane[:, None] * $(512)) + r233_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r233_lane[:, None] * $(512)) + r233_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r233_value = tl.dot(r233_left, r233_right)
      r233_value = r233_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_63_ptr) + ((r233_group * $(49152)) + (r233_row[:, None] * $(512)) + r233_col[None, :]), r233_value)
    }
    r234_z = tl.arange($(49152))
    r234_value = (tl.load($(candidate_vbmm_62_ptr) + ((((r234_z / $(512)) % $(96)) * $(512)) + (r234_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_63_ptr) + ((((r234_z / $(512)) % $(96)) * $(512)) + (r234_z % $(512))), dtype=tl.bfloat16))
    r234_value = r234_value.to(tl.bfloat16)
    r234_value = tl.broadcast_to(r234_value, [$(49152)])
    tl.store($(candidate_vadd_36_ptr) + r234_z, r234_value)
    r235_z = tl.arange($(147456))
    r235_value = tl.where(((((r235_z / $(512)) % $(288)) >= $(192)) & (((r235_z / $(512)) % $(288)) < $(288)) & ((((r235_z / $(512)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vadd_36_ptr) + ((((((r235_z / $(512)) % $(288)) - $(192)) % $(96)) * $(512)) + (r235_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r235_value = r235_value.to(tl.bfloat16)
    r235_value = tl.broadcast_to(r235_value, [$(147456)])
    tl.store($(candidate_vslice_backward_8_ptr) + r235_z, r235_value)
    r236_z = tl.arange($(147456))
    r236_value = (tl.load($(candidate_vadd_30_ptr) + ((((r236_z / $(512)) % $(288)) * $(512)) + (r236_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_8_ptr) + ((((r236_z / $(512)) % $(288)) * $(512)) + (r236_z % $(512))), dtype=tl.bfloat16))
    r236_value = r236_value.to(tl.bfloat16)
    r236_value = tl.broadcast_to(r236_value, [$(147456)])
    tl.store($(candidate_vadd_37_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(18432))
    r237_value = tl.where(((((r237_z / $(64)) % $(288)) >= $(192)) & (((r237_z / $(64)) % $(288)) < $(288)) & ((((r237_z / $(64)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vsum_6_ptr) + ((((((r237_z / $(64)) % $(288)) - $(192)) % $(96)) * $(64)) + (r237_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r237_value = r237_value.to(tl.bfloat16)
    r237_value = tl.broadcast_to(r237_value, [$(18432)])
    tl.store($(candidate_vslice_backward_9_ptr) + r237_z, r237_value)
    r238_z = tl.arange($(18432))
    r238_value = (tl.load($(candidate_vadd_31_ptr) + ((((r238_z / $(64)) % $(288)) * $(64)) + (r238_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_9_ptr) + ((((r238_z / $(64)) % $(288)) * $(64)) + (r238_z % $(64))), dtype=tl.bfloat16))
    r238_value = r238_value.to(tl.bfloat16)
    r238_value = tl.broadcast_to(r238_value, [$(18432)])
    tl.store($(candidate_vadd_38_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(294912))
    r239_value = tl.where(((((r239_z / $(1024)) % $(288)) >= $(192)) & (((r239_z / $(1024)) % $(288)) < $(224)) & ((((r239_z / $(1024)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vcat_42_ptr) + ((((((r239_z / $(1024)) % $(288)) - $(192)) % $(32)) * $(1536)) + (((r239_z / $(128)) % $(8)) * $(192)) + (r239_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r239_value = r239_value.to(tl.bfloat16)
    r239_value = tl.broadcast_to(r239_value, [$(294912)])
    tl.store($(candidate_vslice_backward_10_ptr) + r239_z, r239_value)
    r240_z = tl.arange($(294912))
    r240_value = (tl.load($(candidate_vadd_32_ptr) + ((((r240_z / $(1024)) % $(288)) * $(1024)) + (((r240_z / $(128)) % $(8)) * $(128)) + (r240_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_10_ptr) + ((((r240_z / $(1024)) % $(288)) * $(1024)) + (((r240_z / $(128)) % $(8)) * $(128)) + (r240_z % $(128))), dtype=tl.bfloat16))
    r240_value = r240_value.to(tl.bfloat16)
    r240_value = tl.broadcast_to(r240_value, [$(294912)])
    tl.store($(candidate_vadd_39_ptr) + r240_z, r240_value)
    r241_z = tl.arange($(147456))
    r241_value = tl.where(((((r241_z / $(512)) % $(288)) >= $(192)) & (((r241_z / $(512)) % $(288)) < $(224)) & ((((r241_z / $(512)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vcat_42_ptr) + ((((((r241_z / $(512)) % $(288)) - $(192)) % $(32)) * $(1536)) + (((r241_z / $(64)) % $(8)) * $(192)) + (r241_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r241_value = r241_value.to(tl.bfloat16)
    r241_value = tl.broadcast_to(r241_value, [$(147456)])
    tl.store($(candidate_vslice_backward_11_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(147456))
    r242_value = (tl.load($(candidate_vadd_33_ptr) + ((((r242_z / $(512)) % $(288)) * $(512)) + (((r242_z / $(64)) % $(8)) * $(64)) + (r242_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_11_ptr) + ((((r242_z / $(512)) % $(288)) * $(512)) + (((r242_z / $(64)) % $(8)) * $(64)) + (r242_z % $(64))), dtype=tl.bfloat16))
    r242_value = r242_value.to(tl.bfloat16)
    r242_value = tl.broadcast_to(r242_value, [$(147456)])
    tl.store($(candidate_vadd_40_ptr) + r242_z, r242_value)
    r243_row = tl.arange($(96))
    r243_col = tl.arange($(1024))
    r243_lane = tl.arange($(512))
    for r243_group in range($(1)) {
      r243_left = tl.load($(candidate_vcat_7_ptr) + ((((((r243_row[:, None] * $(512)) + r243_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r243_row[:, None] * $(512)) + r243_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r243_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r243_lane[:, None] * $(1024)) + r243_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r243_lane[:, None] * $(1024)) + r243_col[None, :]) % $(128)) * $(512)) + ((((r243_lane[:, None] * $(1024)) + r243_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r243_value = tl.dot(r243_left, r243_right)
      r243_value = r243_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_66_ptr) + ((r243_group * $(98304)) + (r243_row[:, None] * $(1024)) + r243_col[None, :]), r243_value)
    }
    r244_row = tl.arange($(96))
    r244_col = tl.arange($(1024))
    r244_lane = tl.arange($(512))
    for r244_group in range($(1)) {
      r244_left = tl.load($(candidate_vcat_7_ptr) + ((((((r244_row[:, None] * $(512)) + r244_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r244_row[:, None] * $(512)) + r244_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r244_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r244_lane[:, None] * $(1024)) + r244_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r244_lane[:, None] * $(1024)) + r244_col[None, :]) % $(128)) * $(512)) + ((((r244_lane[:, None] * $(1024)) + r244_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r244_value = tl.dot(r244_left, r244_right)
      r244_value = r244_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_67_ptr) + ((r244_group * $(98304)) + (r244_row[:, None] * $(1024)) + r244_col[None, :]), r244_value)
    }
    r245_z = tl.arange($(49152))
    r245_value = tl.where((r245_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r245_z / $(1536)) % $(32)) * $(1024)) + (((r245_z / $(192)) % $(8)) * $(128)) + ((r245_z % $(192)) - $(0)) + $(163840)) / $(1024)) % $(288)) * $(1536)) + (((((((r245_z / $(1536)) % $(32)) * $(1024)) + (((r245_z / $(192)) % $(8)) * $(128)) + ((r245_z % $(192)) - $(0)) + $(163840)) / $(128)) % $(8)) * $(192)) + (((((r245_z / $(1536)) % $(32)) * $(1024)) + (((r245_z / $(192)) % $(8)) * $(128)) + ((r245_z % $(192)) - $(0)) + $(163840)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r245_z / $(1536)) % $(32)) * $(512)) + (((r245_z / $(192)) % $(8)) * $(64)) + ((r245_z % $(192)) - $(128)) + $(81920)) / $(64)) % $(8)) * $(18432)) + (((((((r245_z / $(1536)) % $(32)) * $(512)) + (((r245_z / $(192)) % $(8)) * $(64)) + ((r245_z % $(192)) - $(128)) + $(81920)) / $(512)) % $(288)) * $(64)) + (((((r245_z / $(1536)) % $(32)) * $(512)) + (((r245_z / $(192)) % $(8)) * $(64)) + ((r245_z % $(192)) - $(128)) + $(81920)) % $(64))), dtype=tl.bfloat16))
    r245_value = r245_value.to(tl.bfloat16)
    r245_value = tl.broadcast_to(r245_value, [$(49152)])
    tl.store($(candidate_vcat_46_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(147456))
    r246_value = tl.where((r246_z % $(192)) < $(128), tl.load($(candidate_vbmm_66_ptr) + ((((((((r246_z / $(1536)) % $(96)) * $(1024)) + (((r246_z / $(192)) % $(8)) * $(128)) + ((r246_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r246_z / $(1536)) % $(96)) * $(1024)) + (((r246_z / $(192)) % $(8)) * $(128)) + ((r246_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r246_z / $(1536)) % $(96)) * $(1024)) + (((r246_z / $(192)) % $(8)) * $(128)) + ((r246_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r246_z / $(1536)) % $(96)) * $(576)) + ((((r246_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r246_value = r246_value.to(tl.bfloat16)
    r246_value = tl.broadcast_to(r246_value, [$(147456)])
    tl.store($(candidate_vcat_47_ptr) + r246_z, r246_value)
    r247_z = tl.arange($(49152))
    r247_value = tl.load($(candidate_vcat_46_ptr) + ((((r247_z / $(192)) % $(32)) * $(1536)) + (((r247_z / $(6144)) % $(8)) * $(192)) + (r247_z % $(192))), dtype=tl.bfloat16)
    r247_value = r247_value.to(tl.float32)
    r247_value = tl.broadcast_to(r247_value, [$(49152)])
    tl.store($(candidate_vto_copy_100_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(147456))
    r248_value = tl.load($(candidate_vcat_47_ptr) + ((((r248_z / $(192)) % $(96)) * $(1536)) + (((r248_z / $(18432)) % $(8)) * $(192)) + (r248_z % $(192))), dtype=tl.bfloat16)
    r248_value = r248_value.to(tl.float32)
    r248_value = tl.broadcast_to(r248_value, [$(147456)])
    tl.store($(candidate_vto_copy_101_ptr) + r248_z, r248_value)
    r249_row = tl.arange($(32))
    r249_col = tl.arange($(96))
    r249_lane = tl.arange($(192))
    for r249_group in range($(8)) {
      r249_left = tl.load($(candidate_vto_copy_100_ptr) + (((((((r249_group % $(8)) * $(6144)) + (r249_row[:, None] * $(192)) + r249_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r249_group % $(8)) * $(6144)) + (r249_row[:, None] * $(192)) + r249_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r249_group % $(8)) * $(6144)) + (r249_row[:, None] * $(192)) + r249_lane[None, :]) % $(192))), dtype=tl.float32)
      r249_right = tl.load($(candidate_vto_copy_101_ptr) + (((((((r249_group % $(8)) * $(18432)) + (r249_lane[:, None] * $(96)) + r249_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r249_group % $(8)) * $(18432)) + (r249_lane[:, None] * $(96)) + r249_col[None, :]) % $(96)) * $(192)) + (((((r249_group % $(8)) * $(18432)) + (r249_lane[:, None] * $(96)) + r249_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r249_value = tl.dot(r249_left, r249_right)
      r249_value = r249_value.to(tl.float32)
      tl.store($(candidate_vbmm_68_ptr) + ((r249_group * $(3072)) + (r249_row[:, None] * $(96)) + r249_col[None, :]), r249_value)
    }
    r250_z = tl.arange($(24576))
    r250_value = (tl.load($(candidate_vbmm_68_ptr) + ((((r250_z / $(3072)) % $(8)) * $(3072)) + (((r250_z / $(96)) % $(32)) * $(96)) + (r250_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r250_value = r250_value.to(tl.float32)
    r250_value = tl.broadcast_to(r250_value, [$(24576)])
    tl.store($(candidate_vmul_48_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(32))
    r251_value = (tl.cast(r251_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r251_value = tl.broadcast_to(r251_value, [$(32)])
    tl.store($(candidate_vadd_41_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(32))
    r252_value = (tl.load($(candidate_vadd_41_ptr) + r252_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r252_value = tl.broadcast_to(r252_value, [$(32)])
    tl.store($(candidate_vsub_36_ptr) + r252_z, r252_value)
    r253_z = tl.arange($(3072))
    r253_value = (tl.cast((r253_z % $(96)), tl.int64) > tl.load($(candidate_vsub_36_ptr) + ((r253_z / $(96)) % $(32)), dtype=tl.int64))
    r253_value = tl.broadcast_to(r253_value, [$(3072)])
    tl.store($(candidate_vgt_15_ptr) + r253_z, r253_value)
    r254_z = tl.arange($(24576))
    r254_value = tl.where(tl.load($(candidate_vgt_15_ptr) + ((((r254_z / $(96)) % $(32)) * $(96)) + (r254_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_48_ptr) + ((((r254_z / $(3072)) % $(8)) * $(3072)) + (((r254_z / $(96)) % $(32)) * $(96)) + (r254_z % $(96))), dtype=tl.float32))
    r254_value = r254_value.to(tl.float32)
    r254_value = tl.broadcast_to(r254_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_15_ptr) + r254_z, r254_value)
    r255_z = tl.arange($(24576))
    r255_lane = tl.arange($(96))
    r255_row_values = tl.load($(candidate_vmasked_fill_15_ptr) + ((((r255_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r255_z[:, None] / $(96)) % $(32)) * $(96)) + (r255_lane[None, :] % $(96))), dtype=tl.float32)
    r255_maximum = tl.max(r255_row_values, axis=1, keep_dims=true)
    r255_weights = tl.exp(r255_row_values - r255_maximum)
    r255_total = tl.sum(r255_weights, axis=1)
    r255_maximum = tl.max(r255_row_values, axis=1)
    r255_value = tl.exp(tl.load($(candidate_vmasked_fill_15_ptr) + ((((r255_z / $(3072)) % $(8)) * $(3072)) + (((r255_z / $(96)) % $(32)) * $(96)) + (r255_z % $(96))), dtype=tl.float32) - r255_maximum) / r255_total
    r255_value = r255_value.to(tl.float32)
    r255_value = tl.broadcast_to(r255_value, [$(24576)])
    tl.store($(candidate_vsoftmax_12_ptr) + r255_z, r255_value)
    r256_z = tl.arange($(24576))
    r256_value = tl.load($(candidate_vsoftmax_12_ptr) + ((((r256_z / $(3072)) % $(8)) * $(3072)) + (((r256_z / $(96)) % $(32)) * $(96)) + (r256_z % $(96))), dtype=tl.float32)
    r256_value = r256_value.to(tl.bfloat16)
    r256_value = tl.broadcast_to(r256_value, [$(24576)])
    tl.store($(candidate_vto_copy_102_ptr) + r256_z, r256_value)
    r257_row = tl.arange($(32))
    r257_col = tl.arange($(128))
    r257_lane = tl.arange($(96))
    for r257_group in range($(8)) {
      r257_left = tl.load($(candidate_vto_copy_102_ptr) + (((((((r257_group % $(8)) * $(3072)) + (r257_row[:, None] * $(96)) + r257_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r257_group % $(8)) * $(3072)) + (r257_row[:, None] * $(96)) + r257_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r257_group % $(8)) * $(3072)) + (r257_row[:, None] * $(96)) + r257_lane[None, :]) % $(96))), dtype=tl.bfloat16)
      r257_right = tl.load($(candidate_vbmm_67_ptr) + (((((((r257_group % $(8)) * $(12288)) + (r257_lane[:, None] * $(128)) + r257_col[None, :]) / $(128)) % $(96)) * $(1024)) + ((((((r257_group % $(8)) * $(12288)) + (r257_lane[:, None] * $(128)) + r257_col[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r257_group % $(8)) * $(12288)) + (r257_lane[:, None] * $(128)) + r257_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r257_value = tl.dot(r257_left, r257_right)
      r257_value = r257_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_69_ptr) + ((r257_group * $(4096)) + (r257_row[:, None] * $(128)) + r257_col[None, :]), r257_value)
    }
    r258_z = tl.arange($(256))
    r258_lane = tl.arange($(96))
    r258_row_values = tl.load($(candidate_vmasked_fill_15_ptr) + ((((r258_z[:, None] / $(32)) % $(8)) * $(3072)) + ((r258_z[:, None] % $(32)) * $(96)) + (r258_lane[None, :] % $(96))), dtype=tl.float32)
    r258_maximum = tl.max(r258_row_values, axis=1, keep_dims=true)
    r258_weights = tl.exp(r258_row_values - r258_maximum)
    r258_total = tl.sum(r258_weights, axis=1)
    r258_maximum = tl.max(r258_row_values, axis=1)
    r258_value = r258_maximum + tl.log(r258_total)
    r258_value = r258_value.to(tl.float32)
    r258_value = tl.broadcast_to(r258_value, [$(256)])
    tl.store($(candidate_vlogsumexp_12_ptr) + r258_z, r258_value)
    r259_z = tl.arange($(32768))
    r259_value = tl.load($(candidate_vbmm_69_ptr) + ((((((((r259_z / $(128)) % $(8)) * $(4096)) + ((((r259_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r259_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r259_z / $(128)) % $(8)) * $(32)) + (((r259_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r259_z % $(128))), dtype=tl.bfloat16)
    r259_value = r259_value.to(tl.bfloat16)
    r259_value = tl.broadcast_to(r259_value, [$(32768)])
    tl.store($(candidate_vcat_48_ptr) + r259_z, r259_value)
    r260_z = tl.arange($(256))
    r260_value = tl.load($(candidate_vlogsumexp_12_ptr) + (((r260_z % $(8)) * $(32)) + ((((r260_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r260_value = r260_value.to(tl.float32)
    r260_value = tl.broadcast_to(r260_value, [$(256)])
    tl.store($(candidate_vcat_49_ptr) + r260_z, r260_value)
    r261_z = tl.arange($(49152))
    r261_value = tl.load($(candidate_vcat_46_ptr) + ((((r261_z / $(192)) % $(32)) * $(1536)) + (((r261_z / $(6144)) % $(8)) * $(192)) + (r261_z % $(192))), dtype=tl.bfloat16)
    r261_value = r261_value.to(tl.float32)
    r261_value = tl.broadcast_to(r261_value, [$(49152)])
    tl.store($(candidate_vto_copy_103_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(147456))
    r262_value = tl.load($(candidate_vcat_47_ptr) + ((((r262_z / $(192)) % $(96)) * $(1536)) + (((r262_z / $(18432)) % $(8)) * $(192)) + (r262_z % $(192))), dtype=tl.bfloat16)
    r262_value = r262_value.to(tl.float32)
    r262_value = tl.broadcast_to(r262_value, [$(147456)])
    tl.store($(candidate_vto_copy_104_ptr) + r262_z, r262_value)
    r263_z = tl.arange($(98304))
    r263_value = tl.load($(candidate_vbmm_67_ptr) + ((((r263_z / $(128)) % $(96)) * $(1024)) + (((r263_z / $(12288)) % $(8)) * $(128)) + (r263_z % $(128))), dtype=tl.bfloat16)
    r263_value = r263_value.to(tl.float32)
    r263_value = tl.broadcast_to(r263_value, [$(98304)])
    tl.store($(candidate_vto_copy_105_ptr) + r263_z, r263_value)
    r264_z = tl.arange($(32768))
    r264_value = tl.load($(candidate_vmm_22_ptr) + ((((r264_z / $(128)) % $(32)) * $(1024)) + (((r264_z / $(4096)) % $(8)) * $(128)) + (r264_z % $(128)) + $(163840)), dtype=tl.bfloat16)
    r264_value = r264_value.to(tl.float32)
    r264_value = tl.broadcast_to(r264_value, [$(32768)])
    tl.store($(candidate_vto_copy_106_ptr) + r264_z, r264_value)
    r265_z = tl.arange($(32768))
    r265_value = tl.load($(candidate_vcat_48_ptr) + ((((r265_z / $(128)) % $(32)) * $(1024)) + (((r265_z / $(4096)) % $(8)) * $(128)) + (r265_z % $(128))), dtype=tl.bfloat16)
    r265_value = r265_value.to(tl.float32)
    r265_value = tl.broadcast_to(r265_value, [$(32768)])
    tl.store($(candidate_vto_copy_107_ptr) + r265_z, r265_value)
    r266_row = tl.arange($(32))
    r266_col = tl.arange($(96))
    r266_lane = tl.arange($(192))
    for r266_group in range($(8)) {
      r266_left = tl.load($(candidate_vto_copy_103_ptr) + (((((((r266_group % $(8)) * $(6144)) + (r266_row[:, None] * $(192)) + r266_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r266_group % $(8)) * $(6144)) + (r266_row[:, None] * $(192)) + r266_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r266_group % $(8)) * $(6144)) + (r266_row[:, None] * $(192)) + r266_lane[None, :]) % $(192))), dtype=tl.float32)
      r266_right = tl.load($(candidate_vto_copy_104_ptr) + (((((((r266_group % $(8)) * $(18432)) + (r266_lane[:, None] * $(96)) + r266_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r266_group % $(8)) * $(18432)) + (r266_lane[:, None] * $(96)) + r266_col[None, :]) % $(96)) * $(192)) + (((((r266_group % $(8)) * $(18432)) + (r266_lane[:, None] * $(96)) + r266_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r266_value = tl.dot(r266_left, r266_right)
      r266_value = r266_value.to(tl.float32)
      tl.store($(candidate_vbmm_70_ptr) + ((r266_group * $(3072)) + (r266_row[:, None] * $(96)) + r266_col[None, :]), r266_value)
    }
    r267_z = tl.arange($(24576))
    r267_value = (tl.load($(candidate_vbmm_70_ptr) + ((((r267_z / $(3072)) % $(8)) * $(3072)) + (((r267_z / $(96)) % $(32)) * $(96)) + (r267_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r267_value = r267_value.to(tl.float32)
    r267_value = tl.broadcast_to(r267_value, [$(24576)])
    tl.store($(candidate_vmul_49_ptr) + r267_z, r267_value)
    r268_z = tl.arange($(32))
    r268_value = (tl.cast(r268_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r268_value = tl.broadcast_to(r268_value, [$(32)])
    tl.store($(candidate_vadd_42_ptr) + r268_z, r268_value)
    r269_z = tl.arange($(32))
    r269_value = (tl.load($(candidate_vadd_42_ptr) + r269_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r269_value = tl.broadcast_to(r269_value, [$(32)])
    tl.store($(candidate_vsub_37_ptr) + r269_z, r269_value)
    r270_z = tl.arange($(3072))
    r270_value = (tl.cast((r270_z % $(96)), tl.int64) > tl.load($(candidate_vsub_37_ptr) + ((r270_z / $(96)) % $(32)), dtype=tl.int64))
    r270_value = tl.broadcast_to(r270_value, [$(3072)])
    tl.store($(candidate_vgt_16_ptr) + r270_z, r270_value)
    r271_z = tl.arange($(24576))
    r271_value = tl.where(tl.load($(candidate_vgt_16_ptr) + ((((r271_z / $(96)) % $(32)) * $(96)) + (r271_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_49_ptr) + ((((r271_z / $(3072)) % $(8)) * $(3072)) + (((r271_z / $(96)) % $(32)) * $(96)) + (r271_z % $(96))), dtype=tl.float32))
    r271_value = r271_value.to(tl.float32)
    r271_value = tl.broadcast_to(r271_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_16_ptr) + r271_z, r271_value)
    r272_z = tl.arange($(24576))
    r272_value = (tl.load($(candidate_vmasked_fill_16_ptr) + ((((r272_z / $(3072)) % $(8)) * $(3072)) + (((r272_z / $(96)) % $(32)) * $(96)) + (r272_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vcat_49_ptr) + ((((r272_z / $(96)) % $(32)) * $(8)) + ((r272_z / $(3072)) % $(8))), dtype=tl.float32))
    r272_value = r272_value.to(tl.float32)
    r272_value = tl.broadcast_to(r272_value, [$(24576)])
    tl.store($(candidate_vsub_38_ptr) + r272_z, r272_value)
    r273_z = tl.arange($(24576))
    r273_value = tl.exp(tl.load($(candidate_vsub_38_ptr) + ((((r273_z / $(3072)) % $(8)) * $(3072)) + (((r273_z / $(96)) % $(32)) * $(96)) + (r273_z % $(96))), dtype=tl.float32))
    r273_value = r273_value.to(tl.float32)
    r273_value = tl.broadcast_to(r273_value, [$(24576)])
    tl.store($(candidate_vexp_3_ptr) + r273_z, r273_value)
    r274_z = tl.arange($(32768))
    r274_value = (tl.load($(candidate_vto_copy_106_ptr) + ((((r274_z / $(4096)) % $(8)) * $(4096)) + (((r274_z / $(128)) % $(32)) * $(128)) + (r274_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_107_ptr) + ((((r274_z / $(4096)) % $(8)) * $(4096)) + (((r274_z / $(128)) % $(32)) * $(128)) + (r274_z % $(128))), dtype=tl.float32))
    r274_value = r274_value.to(tl.float32)
    r274_value = tl.broadcast_to(r274_value, [$(32768)])
    tl.store($(candidate_vmul_50_ptr) + r274_z, r274_value)
    r275_z = tl.arange($(256))
    r275_lane = tl.arange($(128))
    r275_value = tl.sum(tl.load($(candidate_vmul_50_ptr) + ((((r275_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r275_z[:, None] % $(32)) * $(128)) + r275_lane[None, :]), dtype=tl.float32), axis=1)
    r275_value = r275_value.to(tl.float32)
    r275_value = tl.broadcast_to(r275_value, [$(256)])
    tl.store($(candidate_vsum_7_ptr) + r275_z, r275_value)
    r276_row = tl.arange($(32))
    r276_col = tl.arange($(96))
    r276_lane = tl.arange($(128))
    for r276_group in range($(8)) {
      r276_left = tl.load($(candidate_vto_copy_106_ptr) + (((((((r276_group % $(8)) * $(4096)) + (r276_row[:, None] * $(128)) + r276_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r276_group % $(8)) * $(4096)) + (r276_row[:, None] * $(128)) + r276_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r276_group % $(8)) * $(4096)) + (r276_row[:, None] * $(128)) + r276_lane[None, :]) % $(128))), dtype=tl.float32)
      r276_right = tl.load($(candidate_vto_copy_105_ptr) + (((((((r276_group % $(8)) * $(12288)) + (r276_lane[:, None] * $(96)) + r276_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r276_group % $(8)) * $(12288)) + (r276_lane[:, None] * $(96)) + r276_col[None, :]) % $(96)) * $(128)) + (((((r276_group % $(8)) * $(12288)) + (r276_lane[:, None] * $(96)) + r276_col[None, :]) / $(96)) % $(128))), dtype=tl.float32)
      r276_value = tl.dot(r276_left, r276_right)
      r276_value = r276_value.to(tl.float32)
      tl.store($(candidate_vbmm_71_ptr) + ((r276_group * $(3072)) + (r276_row[:, None] * $(96)) + r276_col[None, :]), r276_value)
    }
    r277_z = tl.arange($(24576))
    r277_value = (tl.load($(candidate_vbmm_71_ptr) + ((((r277_z / $(3072)) % $(8)) * $(3072)) + (((r277_z / $(96)) % $(32)) * $(96)) + (r277_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vsum_7_ptr) + ((((r277_z / $(3072)) % $(8)) * $(32)) + ((r277_z / $(96)) % $(32))), dtype=tl.float32))
    r277_value = r277_value.to(tl.float32)
    r277_value = tl.broadcast_to(r277_value, [$(24576)])
    tl.store($(candidate_vsub_39_ptr) + r277_z, r277_value)
    r278_z = tl.arange($(24576))
    r278_value = (tl.load($(candidate_vexp_3_ptr) + ((((r278_z / $(3072)) % $(8)) * $(3072)) + (((r278_z / $(96)) % $(32)) * $(96)) + (r278_z % $(96))), dtype=tl.float32) * tl.load($(candidate_vsub_39_ptr) + ((((r278_z / $(3072)) % $(8)) * $(3072)) + (((r278_z / $(96)) % $(32)) * $(96)) + (r278_z % $(96))), dtype=tl.float32))
    r278_value = r278_value.to(tl.float32)
    r278_value = tl.broadcast_to(r278_value, [$(24576)])
    tl.store($(candidate_vmul_51_ptr) + r278_z, r278_value)
    r279_row = tl.arange($(32))
    r279_col = tl.arange($(192))
    r279_lane = tl.arange($(96))
    for r279_group in range($(8)) {
      r279_left = tl.load($(candidate_vmul_51_ptr) + (((((((r279_group % $(8)) * $(3072)) + (r279_row[:, None] * $(96)) + r279_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r279_group % $(8)) * $(3072)) + (r279_row[:, None] * $(96)) + r279_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r279_group % $(8)) * $(3072)) + (r279_row[:, None] * $(96)) + r279_lane[None, :]) % $(96))), dtype=tl.float32)
      r279_right = tl.load($(candidate_vto_copy_104_ptr) + (((((((r279_group % $(8)) * $(18432)) + (r279_lane[:, None] * $(192)) + r279_col[None, :]) / $(18432)) % $(8)) * $(18432)) + ((((((r279_group % $(8)) * $(18432)) + (r279_lane[:, None] * $(192)) + r279_col[None, :]) / $(192)) % $(96)) * $(192)) + ((((r279_group % $(8)) * $(18432)) + (r279_lane[:, None] * $(192)) + r279_col[None, :]) % $(192))), dtype=tl.float32)
      r279_value = tl.dot(r279_left, r279_right)
      r279_value = r279_value.to(tl.float32)
      tl.store($(candidate_vbmm_72_ptr) + ((r279_group * $(6144)) + (r279_row[:, None] * $(192)) + r279_col[None, :]), r279_value)
    }
    r280_z = tl.arange($(49152))
    r280_value = (tl.load($(candidate_vbmm_72_ptr) + ((((r280_z / $(6144)) % $(8)) * $(6144)) + (((r280_z / $(192)) % $(32)) * $(192)) + (r280_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r280_value = r280_value.to(tl.float32)
    r280_value = tl.broadcast_to(r280_value, [$(49152)])
    tl.store($(candidate_vmul_52_ptr) + r280_z, r280_value)
    r281_z = tl.arange($(49152))
    r281_value = tl.load($(candidate_vmul_52_ptr) + ((((r281_z / $(192)) % $(8)) * $(6144)) + (((r281_z / $(1536)) % $(32)) * $(192)) + (r281_z % $(192))), dtype=tl.float32)
    r281_value = r281_value.to(tl.bfloat16)
    r281_value = tl.broadcast_to(r281_value, [$(49152)])
    tl.store($(candidate_vto_copy_108_ptr) + r281_z, r281_value)
    r282_row = tl.arange($(96))
    r282_col = tl.arange($(192))
    r282_lane = tl.arange($(32))
    for r282_group in range($(8)) {
      r282_left = tl.load($(candidate_vmul_51_ptr) + (((((((r282_group % $(8)) * $(3072)) + (r282_row[:, None] * $(32)) + r282_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r282_group % $(8)) * $(3072)) + (r282_row[:, None] * $(32)) + r282_lane[None, :]) % $(32)) * $(96)) + (((((r282_group % $(8)) * $(3072)) + (r282_row[:, None] * $(32)) + r282_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r282_right = tl.load($(candidate_vto_copy_103_ptr) + (((((((r282_group % $(8)) * $(6144)) + (r282_lane[:, None] * $(192)) + r282_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r282_group % $(8)) * $(6144)) + (r282_lane[:, None] * $(192)) + r282_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r282_group % $(8)) * $(6144)) + (r282_lane[:, None] * $(192)) + r282_col[None, :]) % $(192))), dtype=tl.float32)
      r282_value = tl.dot(r282_left, r282_right)
      r282_value = r282_value.to(tl.float32)
      tl.store($(candidate_vbmm_73_ptr) + ((r282_group * $(18432)) + (r282_row[:, None] * $(192)) + r282_col[None, :]), r282_value)
    }
    r283_z = tl.arange($(147456))
    r283_value = (tl.load($(candidate_vbmm_73_ptr) + ((((r283_z / $(18432)) % $(8)) * $(18432)) + (((r283_z / $(192)) % $(96)) * $(192)) + (r283_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r283_value = r283_value.to(tl.float32)
    r283_value = tl.broadcast_to(r283_value, [$(147456)])
    tl.store($(candidate_vmul_53_ptr) + r283_z, r283_value)
    r284_z = tl.arange($(147456))
    r284_value = tl.load($(candidate_vmul_53_ptr) + ((((r284_z / $(192)) % $(8)) * $(18432)) + (((r284_z / $(1536)) % $(96)) * $(192)) + (r284_z % $(192))), dtype=tl.float32)
    r284_value = r284_value.to(tl.bfloat16)
    r284_value = tl.broadcast_to(r284_value, [$(147456)])
    tl.store($(candidate_vto_copy_109_ptr) + r284_z, r284_value)
    r285_row = tl.arange($(96))
    r285_col = tl.arange($(128))
    r285_lane = tl.arange($(32))
    for r285_group in range($(8)) {
      r285_left = tl.load($(candidate_vexp_3_ptr) + (((((((r285_group % $(8)) * $(3072)) + (r285_row[:, None] * $(32)) + r285_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r285_group % $(8)) * $(3072)) + (r285_row[:, None] * $(32)) + r285_lane[None, :]) % $(32)) * $(96)) + (((((r285_group % $(8)) * $(3072)) + (r285_row[:, None] * $(32)) + r285_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r285_right = tl.load($(candidate_vto_copy_106_ptr) + (((((((r285_group % $(8)) * $(4096)) + (r285_lane[:, None] * $(128)) + r285_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r285_group % $(8)) * $(4096)) + (r285_lane[:, None] * $(128)) + r285_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r285_group % $(8)) * $(4096)) + (r285_lane[:, None] * $(128)) + r285_col[None, :]) % $(128))), dtype=tl.float32)
      r285_value = tl.dot(r285_left, r285_right)
      r285_value = r285_value.to(tl.float32)
      tl.store($(candidate_vbmm_74_ptr) + ((r285_group * $(12288)) + (r285_row[:, None] * $(128)) + r285_col[None, :]), r285_value)
    }
    r286_z = tl.arange($(98304))
    r286_value = tl.load($(candidate_vbmm_74_ptr) + ((((r286_z / $(128)) % $(8)) * $(12288)) + (((r286_z / $(1024)) % $(96)) * $(128)) + (r286_z % $(128))), dtype=tl.float32)
    r286_value = r286_value.to(tl.bfloat16)
    r286_value = tl.broadcast_to(r286_value, [$(98304)])
    tl.store($(candidate_vto_copy_110_ptr) + r286_z, r286_value)
    r287_z = tl.arange($(49152))
    r287_value = tl.load($(candidate_vto_copy_108_ptr) + ((((((r287_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(1536)) + (((r287_z / $(192)) % $(8)) * $(192)) + (r287_z % $(192))), dtype=tl.bfloat16)
    r287_value = r287_value.to(tl.bfloat16)
    r287_value = tl.broadcast_to(r287_value, [$(49152)])
    tl.store($(candidate_vcat_50_ptr) + r287_z, r287_value)
    r288_z = tl.arange($(147456))
    r288_value = tl.load($(candidate_vto_copy_109_ptr) + ((((((r288_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r288_z / $(192)) % $(8)) * $(192)) + (r288_z % $(192))), dtype=tl.bfloat16)
    r288_value = r288_value.to(tl.bfloat16)
    r288_value = tl.broadcast_to(r288_value, [$(147456)])
    tl.store($(candidate_vcat_51_ptr) + r288_z, r288_value)
    r289_z = tl.arange($(98304))
    r289_value = tl.load($(candidate_vto_copy_110_ptr) + ((((((r289_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r289_z / $(128)) % $(8)) * $(128)) + (r289_z % $(128))), dtype=tl.bfloat16)
    r289_value = r289_value.to(tl.bfloat16)
    r289_value = tl.broadcast_to(r289_value, [$(98304)])
    tl.store($(candidate_vcat_52_ptr) + r289_z, r289_value)
    r290_z = tl.arange($(6144))
    r290_lane = tl.arange($(8))
    r290_value = tl.sum(tl.load($(candidate_vcat_51_ptr) + ((((r290_z[:, None] / $(64)) % $(96)) * $(1536)) + (r290_lane[None, :] * $(192)) + (r290_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r290_value = r290_value.to(tl.bfloat16)
    r290_value = tl.broadcast_to(r290_value, [$(6144)])
    tl.store($(candidate_vsum_8_ptr) + r290_z, r290_value)
    r291_row = tl.arange($(96))
    r291_col = tl.arange($(512))
    r291_lane = tl.arange($(1024))
    for r291_group in range($(1)) {
      r291_left = tl.load($(candidate_vcat_51_ptr) + ((((((r291_row[:, None] * $(1024)) + r291_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + ((((r291_row[:, None] * $(1024)) + r291_lane[None, :]) % $(8)) * $(192)) + ((((r291_row[:, None] * $(1024)) + r291_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r291_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r291_lane[:, None] * $(512)) + r291_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r291_lane[:, None] * $(512)) + r291_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r291_lane[:, None] * $(512)) + r291_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r291_value = tl.dot(r291_left, r291_right)
      r291_value = r291_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_75_ptr) + ((r291_group * $(49152)) + (r291_row[:, None] * $(512)) + r291_col[None, :]), r291_value)
    }
    r292_row = tl.arange($(96))
    r292_col = tl.arange($(512))
    r292_lane = tl.arange($(1024))
    for r292_group in range($(1)) {
      r292_left = tl.load($(candidate_vcat_52_ptr) + ((((((r292_row[:, None] * $(1024)) + r292_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + ((((r292_row[:, None] * $(1024)) + r292_lane[None, :]) % $(8)) * $(128)) + ((((r292_row[:, None] * $(1024)) + r292_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r292_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r292_lane[:, None] * $(512)) + r292_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r292_lane[:, None] * $(512)) + r292_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r292_lane[:, None] * $(512)) + r292_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r292_value = tl.dot(r292_left, r292_right)
      r292_value = r292_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_76_ptr) + ((r292_group * $(49152)) + (r292_row[:, None] * $(512)) + r292_col[None, :]), r292_value)
    }
    r293_z = tl.arange($(49152))
    r293_value = (tl.load($(candidate_vbmm_75_ptr) + ((((r293_z / $(512)) % $(96)) * $(512)) + (r293_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_76_ptr) + ((((r293_z / $(512)) % $(96)) * $(512)) + (r293_z % $(512))), dtype=tl.bfloat16))
    r293_value = r293_value.to(tl.bfloat16)
    r293_value = tl.broadcast_to(r293_value, [$(49152)])
    tl.store($(candidate_vadd_43_ptr) + r293_z, r293_value)
    r294_z = tl.arange($(147456))
    r294_value = tl.where(((((r294_z / $(512)) % $(288)) >= $(96)) & (((r294_z / $(512)) % $(288)) < $(192)) & ((((r294_z / $(512)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vadd_43_ptr) + ((((((r294_z / $(512)) % $(288)) - $(96)) % $(96)) * $(512)) + (r294_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r294_value = r294_value.to(tl.bfloat16)
    r294_value = tl.broadcast_to(r294_value, [$(147456)])
    tl.store($(candidate_vslice_backward_12_ptr) + r294_z, r294_value)
    r295_z = tl.arange($(147456))
    r295_value = (tl.load($(candidate_vadd_37_ptr) + ((((r295_z / $(512)) % $(288)) * $(512)) + (r295_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_12_ptr) + ((((r295_z / $(512)) % $(288)) * $(512)) + (r295_z % $(512))), dtype=tl.bfloat16))
    r295_value = r295_value.to(tl.bfloat16)
    r295_value = tl.broadcast_to(r295_value, [$(147456)])
    tl.store($(candidate_vadd_44_ptr) + r295_z, r295_value)
    r296_z = tl.arange($(18432))
    r296_value = tl.where(((((r296_z / $(64)) % $(288)) >= $(96)) & (((r296_z / $(64)) % $(288)) < $(192)) & ((((r296_z / $(64)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vsum_8_ptr) + ((((((r296_z / $(64)) % $(288)) - $(96)) % $(96)) * $(64)) + (r296_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r296_value = r296_value.to(tl.bfloat16)
    r296_value = tl.broadcast_to(r296_value, [$(18432)])
    tl.store($(candidate_vslice_backward_13_ptr) + r296_z, r296_value)
    r297_z = tl.arange($(18432))
    r297_value = (tl.load($(candidate_vadd_38_ptr) + ((((r297_z / $(64)) % $(288)) * $(64)) + (r297_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_13_ptr) + ((((r297_z / $(64)) % $(288)) * $(64)) + (r297_z % $(64))), dtype=tl.bfloat16))
    r297_value = r297_value.to(tl.bfloat16)
    r297_value = tl.broadcast_to(r297_value, [$(18432)])
    tl.store($(candidate_vadd_45_ptr) + r297_z, r297_value)
    r298_z = tl.arange($(294912))
    r298_value = tl.where(((((r298_z / $(1024)) % $(288)) >= $(160)) & (((r298_z / $(1024)) % $(288)) < $(192)) & ((((r298_z / $(1024)) % $(288)) - $(160)) % $(1) == $(0))), tl.load($(candidate_vcat_50_ptr) + ((((((r298_z / $(1024)) % $(288)) - $(160)) % $(32)) * $(1536)) + (((r298_z / $(128)) % $(8)) * $(192)) + (r298_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r298_value = r298_value.to(tl.bfloat16)
    r298_value = tl.broadcast_to(r298_value, [$(294912)])
    tl.store($(candidate_vslice_backward_14_ptr) + r298_z, r298_value)
    r299_z = tl.arange($(294912))
    r299_value = (tl.load($(candidate_vadd_39_ptr) + ((((r299_z / $(1024)) % $(288)) * $(1024)) + (((r299_z / $(128)) % $(8)) * $(128)) + (r299_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_14_ptr) + ((((r299_z / $(1024)) % $(288)) * $(1024)) + (((r299_z / $(128)) % $(8)) * $(128)) + (r299_z % $(128))), dtype=tl.bfloat16))
    r299_value = r299_value.to(tl.bfloat16)
    r299_value = tl.broadcast_to(r299_value, [$(294912)])
    tl.store($(candidate_vadd_46_ptr) + r299_z, r299_value)
    r300_z = tl.arange($(147456))
    r300_value = tl.where(((((r300_z / $(512)) % $(288)) >= $(160)) & (((r300_z / $(512)) % $(288)) < $(192)) & ((((r300_z / $(512)) % $(288)) - $(160)) % $(1) == $(0))), tl.load($(candidate_vcat_50_ptr) + ((((((r300_z / $(512)) % $(288)) - $(160)) % $(32)) * $(1536)) + (((r300_z / $(64)) % $(8)) * $(192)) + (r300_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r300_value = r300_value.to(tl.bfloat16)
    r300_value = tl.broadcast_to(r300_value, [$(147456)])
    tl.store($(candidate_vslice_backward_15_ptr) + r300_z, r300_value)
    r301_z = tl.arange($(147456))
    r301_value = (tl.load($(candidate_vadd_40_ptr) + ((((r301_z / $(512)) % $(288)) * $(512)) + (((r301_z / $(64)) % $(8)) * $(64)) + (r301_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_15_ptr) + ((((r301_z / $(512)) % $(288)) * $(512)) + (((r301_z / $(64)) % $(8)) * $(64)) + (r301_z % $(64))), dtype=tl.bfloat16))
    r301_value = r301_value.to(tl.bfloat16)
    r301_value = tl.broadcast_to(r301_value, [$(147456)])
    tl.store($(candidate_vadd_47_ptr) + r301_z, r301_value)
    r302_row = tl.arange($(96))
    r302_col = tl.arange($(1024))
    r302_lane = tl.arange($(512))
    for r302_group in range($(1)) {
      r302_left = tl.load($(candidate_vcat_7_ptr) + ((((((r302_row[:, None] * $(512)) + r302_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r302_row[:, None] * $(512)) + r302_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r302_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r302_lane[:, None] * $(1024)) + r302_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r302_lane[:, None] * $(1024)) + r302_col[None, :]) % $(128)) * $(512)) + ((((r302_lane[:, None] * $(1024)) + r302_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r302_value = tl.dot(r302_left, r302_right)
      r302_value = r302_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_79_ptr) + ((r302_group * $(98304)) + (r302_row[:, None] * $(1024)) + r302_col[None, :]), r302_value)
    }
    r303_row = tl.arange($(96))
    r303_col = tl.arange($(1024))
    r303_lane = tl.arange($(512))
    for r303_group in range($(1)) {
      r303_left = tl.load($(candidate_vcat_7_ptr) + ((((((r303_row[:, None] * $(512)) + r303_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r303_row[:, None] * $(512)) + r303_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r303_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r303_lane[:, None] * $(1024)) + r303_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r303_lane[:, None] * $(1024)) + r303_col[None, :]) % $(128)) * $(512)) + ((((r303_lane[:, None] * $(1024)) + r303_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r303_value = tl.dot(r303_left, r303_right)
      r303_value = r303_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_80_ptr) + ((r303_group * $(98304)) + (r303_row[:, None] * $(1024)) + r303_col[None, :]), r303_value)
    }
    r304_z = tl.arange($(49152))
    r304_value = tl.where((r304_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r304_z / $(1536)) % $(32)) * $(1024)) + (((r304_z / $(192)) % $(8)) * $(128)) + ((r304_z % $(192)) - $(0)) + $(131072)) / $(1024)) % $(288)) * $(1536)) + (((((((r304_z / $(1536)) % $(32)) * $(1024)) + (((r304_z / $(192)) % $(8)) * $(128)) + ((r304_z % $(192)) - $(0)) + $(131072)) / $(128)) % $(8)) * $(192)) + (((((r304_z / $(1536)) % $(32)) * $(1024)) + (((r304_z / $(192)) % $(8)) * $(128)) + ((r304_z % $(192)) - $(0)) + $(131072)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r304_z / $(1536)) % $(32)) * $(512)) + (((r304_z / $(192)) % $(8)) * $(64)) + ((r304_z % $(192)) - $(128)) + $(65536)) / $(64)) % $(8)) * $(18432)) + (((((((r304_z / $(1536)) % $(32)) * $(512)) + (((r304_z / $(192)) % $(8)) * $(64)) + ((r304_z % $(192)) - $(128)) + $(65536)) / $(512)) % $(288)) * $(64)) + (((((r304_z / $(1536)) % $(32)) * $(512)) + (((r304_z / $(192)) % $(8)) * $(64)) + ((r304_z % $(192)) - $(128)) + $(65536)) % $(64))), dtype=tl.bfloat16))
    r304_value = r304_value.to(tl.bfloat16)
    r304_value = tl.broadcast_to(r304_value, [$(49152)])
    tl.store($(candidate_vcat_54_ptr) + r304_z, r304_value)
    r305_z = tl.arange($(147456))
    r305_value = tl.where((r305_z % $(192)) < $(128), tl.load($(candidate_vbmm_79_ptr) + ((((((((r305_z / $(1536)) % $(96)) * $(1024)) + (((r305_z / $(192)) % $(8)) * $(128)) + ((r305_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r305_z / $(1536)) % $(96)) * $(1024)) + (((r305_z / $(192)) % $(8)) * $(128)) + ((r305_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r305_z / $(1536)) % $(96)) * $(1024)) + (((r305_z / $(192)) % $(8)) * $(128)) + ((r305_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r305_z / $(1536)) % $(96)) * $(576)) + ((((r305_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r305_value = r305_value.to(tl.bfloat16)
    r305_value = tl.broadcast_to(r305_value, [$(147456)])
    tl.store($(candidate_vcat_55_ptr) + r305_z, r305_value)
    r306_z = tl.arange($(49152))
    r306_value = tl.load($(candidate_vcat_54_ptr) + ((((r306_z / $(192)) % $(32)) * $(1536)) + (((r306_z / $(6144)) % $(8)) * $(192)) + (r306_z % $(192))), dtype=tl.bfloat16)
    r306_value = r306_value.to(tl.float32)
    r306_value = tl.broadcast_to(r306_value, [$(49152)])
    tl.store($(candidate_vto_copy_111_ptr) + r306_z, r306_value)
    r307_z = tl.arange($(98304))
    r307_value = tl.load($(candidate_vcat_55_ptr) + ((((r307_z / $(192)) % $(64)) * $(1536)) + (((r307_z / $(12288)) % $(8)) * $(192)) + (r307_z % $(192))), dtype=tl.bfloat16)
    r307_value = r307_value.to(tl.float32)
    r307_value = tl.broadcast_to(r307_value, [$(98304)])
    tl.store($(candidate_vto_copy_112_ptr) + r307_z, r307_value)
    r308_row = tl.arange($(32))
    r308_col = tl.arange($(64))
    r308_lane = tl.arange($(192))
    for r308_group in range($(8)) {
      r308_left = tl.load($(candidate_vto_copy_111_ptr) + (((((((r308_group % $(8)) * $(6144)) + (r308_row[:, None] * $(192)) + r308_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r308_group % $(8)) * $(6144)) + (r308_row[:, None] * $(192)) + r308_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r308_group % $(8)) * $(6144)) + (r308_row[:, None] * $(192)) + r308_lane[None, :]) % $(192))), dtype=tl.float32)
      r308_right = tl.load($(candidate_vto_copy_112_ptr) + (((((((r308_group % $(8)) * $(12288)) + (r308_lane[:, None] * $(64)) + r308_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r308_group % $(8)) * $(12288)) + (r308_lane[:, None] * $(64)) + r308_col[None, :]) % $(64)) * $(192)) + (((((r308_group % $(8)) * $(12288)) + (r308_lane[:, None] * $(64)) + r308_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r308_value = tl.dot(r308_left, r308_right)
      r308_value = r308_value.to(tl.float32)
      tl.store($(candidate_vbmm_81_ptr) + ((r308_group * $(2048)) + (r308_row[:, None] * $(64)) + r308_col[None, :]), r308_value)
    }
    r309_z = tl.arange($(16384))
    r309_value = (tl.load($(candidate_vbmm_81_ptr) + ((((r309_z / $(2048)) % $(8)) * $(2048)) + (((r309_z / $(64)) % $(32)) * $(64)) + (r309_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r309_value = r309_value.to(tl.float32)
    r309_value = tl.broadcast_to(r309_value, [$(16384)])
    tl.store($(candidate_vmul_54_ptr) + r309_z, r309_value)
    r310_z = tl.arange($(32))
    r310_value = (tl.cast(r310_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r310_value = tl.broadcast_to(r310_value, [$(32)])
    tl.store($(candidate_vadd_48_ptr) + r310_z, r310_value)
    r311_z = tl.arange($(32))
    r311_value = (tl.load($(candidate_vadd_48_ptr) + r311_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r311_value = tl.broadcast_to(r311_value, [$(32)])
    tl.store($(candidate_vsub_40_ptr) + r311_z, r311_value)
    r312_z = tl.arange($(2048))
    r312_value = (tl.cast((r312_z % $(64)), tl.int64) > tl.load($(candidate_vsub_40_ptr) + ((r312_z / $(64)) % $(32)), dtype=tl.int64))
    r312_value = tl.broadcast_to(r312_value, [$(2048)])
    tl.store($(candidate_vgt_17_ptr) + r312_z, r312_value)
    r313_z = tl.arange($(16384))
    r313_value = tl.where(tl.load($(candidate_vgt_17_ptr) + ((((r313_z / $(64)) % $(32)) * $(64)) + (r313_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_54_ptr) + ((((r313_z / $(2048)) % $(8)) * $(2048)) + (((r313_z / $(64)) % $(32)) * $(64)) + (r313_z % $(64))), dtype=tl.float32))
    r313_value = r313_value.to(tl.float32)
    r313_value = tl.broadcast_to(r313_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_17_ptr) + r313_z, r313_value)
    r314_z = tl.arange($(16384))
    r314_lane = tl.arange($(64))
    r314_row_values = tl.load($(candidate_vmasked_fill_17_ptr) + ((((r314_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r314_z[:, None] / $(64)) % $(32)) * $(64)) + (r314_lane[None, :] % $(64))), dtype=tl.float32)
    r314_maximum = tl.max(r314_row_values, axis=1, keep_dims=true)
    r314_weights = tl.exp(r314_row_values - r314_maximum)
    r314_total = tl.sum(r314_weights, axis=1)
    r314_maximum = tl.max(r314_row_values, axis=1)
    r314_value = tl.exp(tl.load($(candidate_vmasked_fill_17_ptr) + ((((r314_z / $(2048)) % $(8)) * $(2048)) + (((r314_z / $(64)) % $(32)) * $(64)) + (r314_z % $(64))), dtype=tl.float32) - r314_maximum) / r314_total
    r314_value = r314_value.to(tl.float32)
    r314_value = tl.broadcast_to(r314_value, [$(16384)])
    tl.store($(candidate_vsoftmax_13_ptr) + r314_z, r314_value)
    r315_z = tl.arange($(16384))
    r315_value = tl.load($(candidate_vsoftmax_13_ptr) + ((((r315_z / $(2048)) % $(8)) * $(2048)) + (((r315_z / $(64)) % $(32)) * $(64)) + (r315_z % $(64))), dtype=tl.float32)
    r315_value = r315_value.to(tl.bfloat16)
    r315_value = tl.broadcast_to(r315_value, [$(16384)])
    tl.store($(candidate_vto_copy_113_ptr) + r315_z, r315_value)
    r316_row = tl.arange($(32))
    r316_col = tl.arange($(128))
    r316_lane = tl.arange($(64))
    for r316_group in range($(8)) {
      r316_left = tl.load($(candidate_vto_copy_113_ptr) + (((((((r316_group % $(8)) * $(2048)) + (r316_row[:, None] * $(64)) + r316_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r316_group % $(8)) * $(2048)) + (r316_row[:, None] * $(64)) + r316_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r316_group % $(8)) * $(2048)) + (r316_row[:, None] * $(64)) + r316_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r316_right = tl.load($(candidate_vbmm_80_ptr) + (((((((r316_group % $(8)) * $(8192)) + (r316_lane[:, None] * $(128)) + r316_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r316_group % $(8)) * $(8192)) + (r316_lane[:, None] * $(128)) + r316_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r316_group % $(8)) * $(8192)) + (r316_lane[:, None] * $(128)) + r316_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r316_value = tl.dot(r316_left, r316_right)
      r316_value = r316_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_82_ptr) + ((r316_group * $(4096)) + (r316_row[:, None] * $(128)) + r316_col[None, :]), r316_value)
    }
    r317_z = tl.arange($(256))
    r317_lane = tl.arange($(64))
    r317_row_values = tl.load($(candidate_vmasked_fill_17_ptr) + ((((r317_z[:, None] / $(32)) % $(8)) * $(2048)) + ((r317_z[:, None] % $(32)) * $(64)) + (r317_lane[None, :] % $(64))), dtype=tl.float32)
    r317_maximum = tl.max(r317_row_values, axis=1, keep_dims=true)
    r317_weights = tl.exp(r317_row_values - r317_maximum)
    r317_total = tl.sum(r317_weights, axis=1)
    r317_maximum = tl.max(r317_row_values, axis=1)
    r317_value = r317_maximum + tl.log(r317_total)
    r317_value = r317_value.to(tl.float32)
    r317_value = tl.broadcast_to(r317_value, [$(256)])
    tl.store($(candidate_vlogsumexp_13_ptr) + r317_z, r317_value)
    r318_z = tl.arange($(32768))
    r318_value = tl.load($(candidate_vbmm_82_ptr) + ((((((((r318_z / $(128)) % $(8)) * $(4096)) + ((((r318_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r318_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r318_z / $(128)) % $(8)) * $(32)) + (((r318_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r318_z % $(128))), dtype=tl.bfloat16)
    r318_value = r318_value.to(tl.bfloat16)
    r318_value = tl.broadcast_to(r318_value, [$(32768)])
    tl.store($(candidate_vcat_56_ptr) + r318_z, r318_value)
    r319_z = tl.arange($(256))
    r319_value = tl.load($(candidate_vlogsumexp_13_ptr) + (((r319_z % $(8)) * $(32)) + ((((r319_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r319_value = r319_value.to(tl.float32)
    r319_value = tl.broadcast_to(r319_value, [$(256)])
    tl.store($(candidate_vcat_57_ptr) + r319_z, r319_value)
    r320_z = tl.arange($(49152))
    r320_value = tl.load($(candidate_vcat_54_ptr) + ((((r320_z / $(192)) % $(32)) * $(1536)) + (((r320_z / $(6144)) % $(8)) * $(192)) + (r320_z % $(192))), dtype=tl.bfloat16)
    r320_value = r320_value.to(tl.float32)
    r320_value = tl.broadcast_to(r320_value, [$(49152)])
    tl.store($(candidate_vto_copy_114_ptr) + r320_z, r320_value)
    r321_z = tl.arange($(98304))
    r321_value = tl.load($(candidate_vcat_55_ptr) + ((((r321_z / $(192)) % $(64)) * $(1536)) + (((r321_z / $(12288)) % $(8)) * $(192)) + (r321_z % $(192))), dtype=tl.bfloat16)
    r321_value = r321_value.to(tl.float32)
    r321_value = tl.broadcast_to(r321_value, [$(98304)])
    tl.store($(candidate_vto_copy_115_ptr) + r321_z, r321_value)
    r322_z = tl.arange($(65536))
    r322_value = tl.load($(candidate_vbmm_80_ptr) + ((((r322_z / $(128)) % $(64)) * $(1024)) + (((r322_z / $(8192)) % $(8)) * $(128)) + (r322_z % $(128))), dtype=tl.bfloat16)
    r322_value = r322_value.to(tl.float32)
    r322_value = tl.broadcast_to(r322_value, [$(65536)])
    tl.store($(candidate_vto_copy_116_ptr) + r322_z, r322_value)
    r323_z = tl.arange($(32768))
    r323_value = tl.load($(candidate_vmm_22_ptr) + ((((r323_z / $(128)) % $(32)) * $(1024)) + (((r323_z / $(4096)) % $(8)) * $(128)) + (r323_z % $(128)) + $(131072)), dtype=tl.bfloat16)
    r323_value = r323_value.to(tl.float32)
    r323_value = tl.broadcast_to(r323_value, [$(32768)])
    tl.store($(candidate_vto_copy_117_ptr) + r323_z, r323_value)
    r324_z = tl.arange($(32768))
    r324_value = tl.load($(candidate_vcat_56_ptr) + ((((r324_z / $(128)) % $(32)) * $(1024)) + (((r324_z / $(4096)) % $(8)) * $(128)) + (r324_z % $(128))), dtype=tl.bfloat16)
    r324_value = r324_value.to(tl.float32)
    r324_value = tl.broadcast_to(r324_value, [$(32768)])
    tl.store($(candidate_vto_copy_118_ptr) + r324_z, r324_value)
    r325_row = tl.arange($(32))
    r325_col = tl.arange($(64))
    r325_lane = tl.arange($(192))
    for r325_group in range($(8)) {
      r325_left = tl.load($(candidate_vto_copy_114_ptr) + (((((((r325_group % $(8)) * $(6144)) + (r325_row[:, None] * $(192)) + r325_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r325_group % $(8)) * $(6144)) + (r325_row[:, None] * $(192)) + r325_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r325_group % $(8)) * $(6144)) + (r325_row[:, None] * $(192)) + r325_lane[None, :]) % $(192))), dtype=tl.float32)
      r325_right = tl.load($(candidate_vto_copy_115_ptr) + (((((((r325_group % $(8)) * $(12288)) + (r325_lane[:, None] * $(64)) + r325_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r325_group % $(8)) * $(12288)) + (r325_lane[:, None] * $(64)) + r325_col[None, :]) % $(64)) * $(192)) + (((((r325_group % $(8)) * $(12288)) + (r325_lane[:, None] * $(64)) + r325_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r325_value = tl.dot(r325_left, r325_right)
      r325_value = r325_value.to(tl.float32)
      tl.store($(candidate_vbmm_83_ptr) + ((r325_group * $(2048)) + (r325_row[:, None] * $(64)) + r325_col[None, :]), r325_value)
    }
    r326_z = tl.arange($(16384))
    r326_value = (tl.load($(candidate_vbmm_83_ptr) + ((((r326_z / $(2048)) % $(8)) * $(2048)) + (((r326_z / $(64)) % $(32)) * $(64)) + (r326_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r326_value = r326_value.to(tl.float32)
    r326_value = tl.broadcast_to(r326_value, [$(16384)])
    tl.store($(candidate_vmul_55_ptr) + r326_z, r326_value)
    r327_z = tl.arange($(32))
    r327_value = (tl.cast(r327_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r327_value = tl.broadcast_to(r327_value, [$(32)])
    tl.store($(candidate_vadd_49_ptr) + r327_z, r327_value)
    r328_z = tl.arange($(32))
    r328_value = (tl.load($(candidate_vadd_49_ptr) + r328_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r328_value = tl.broadcast_to(r328_value, [$(32)])
    tl.store($(candidate_vsub_41_ptr) + r328_z, r328_value)
    r329_z = tl.arange($(2048))
    r329_value = (tl.cast((r329_z % $(64)), tl.int64) > tl.load($(candidate_vsub_41_ptr) + ((r329_z / $(64)) % $(32)), dtype=tl.int64))
    r329_value = tl.broadcast_to(r329_value, [$(2048)])
    tl.store($(candidate_vgt_18_ptr) + r329_z, r329_value)
    r330_z = tl.arange($(16384))
    r330_value = tl.where(tl.load($(candidate_vgt_18_ptr) + ((((r330_z / $(64)) % $(32)) * $(64)) + (r330_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_55_ptr) + ((((r330_z / $(2048)) % $(8)) * $(2048)) + (((r330_z / $(64)) % $(32)) * $(64)) + (r330_z % $(64))), dtype=tl.float32))
    r330_value = r330_value.to(tl.float32)
    r330_value = tl.broadcast_to(r330_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_18_ptr) + r330_z, r330_value)
    r331_z = tl.arange($(16384))
    r331_value = (tl.load($(candidate_vmasked_fill_18_ptr) + ((((r331_z / $(2048)) % $(8)) * $(2048)) + (((r331_z / $(64)) % $(32)) * $(64)) + (r331_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_57_ptr) + ((((r331_z / $(64)) % $(32)) * $(8)) + ((r331_z / $(2048)) % $(8))), dtype=tl.float32))
    r331_value = r331_value.to(tl.float32)
    r331_value = tl.broadcast_to(r331_value, [$(16384)])
    tl.store($(candidate_vsub_42_ptr) + r331_z, r331_value)
    r332_z = tl.arange($(16384))
    r332_value = tl.exp(tl.load($(candidate_vsub_42_ptr) + ((((r332_z / $(2048)) % $(8)) * $(2048)) + (((r332_z / $(64)) % $(32)) * $(64)) + (r332_z % $(64))), dtype=tl.float32))
    r332_value = r332_value.to(tl.float32)
    r332_value = tl.broadcast_to(r332_value, [$(16384)])
    tl.store($(candidate_vexp_4_ptr) + r332_z, r332_value)
    r333_z = tl.arange($(32768))
    r333_value = (tl.load($(candidate_vto_copy_117_ptr) + ((((r333_z / $(4096)) % $(8)) * $(4096)) + (((r333_z / $(128)) % $(32)) * $(128)) + (r333_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_118_ptr) + ((((r333_z / $(4096)) % $(8)) * $(4096)) + (((r333_z / $(128)) % $(32)) * $(128)) + (r333_z % $(128))), dtype=tl.float32))
    r333_value = r333_value.to(tl.float32)
    r333_value = tl.broadcast_to(r333_value, [$(32768)])
    tl.store($(candidate_vmul_56_ptr) + r333_z, r333_value)
    r334_z = tl.arange($(256))
    r334_lane = tl.arange($(128))
    r334_value = tl.sum(tl.load($(candidate_vmul_56_ptr) + ((((r334_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r334_z[:, None] % $(32)) * $(128)) + r334_lane[None, :]), dtype=tl.float32), axis=1)
    r334_value = r334_value.to(tl.float32)
    r334_value = tl.broadcast_to(r334_value, [$(256)])
    tl.store($(candidate_vsum_9_ptr) + r334_z, r334_value)
    r335_row = tl.arange($(32))
    r335_col = tl.arange($(64))
    r335_lane = tl.arange($(128))
    for r335_group in range($(8)) {
      r335_left = tl.load($(candidate_vto_copy_117_ptr) + (((((((r335_group % $(8)) * $(4096)) + (r335_row[:, None] * $(128)) + r335_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r335_group % $(8)) * $(4096)) + (r335_row[:, None] * $(128)) + r335_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r335_group % $(8)) * $(4096)) + (r335_row[:, None] * $(128)) + r335_lane[None, :]) % $(128))), dtype=tl.float32)
      r335_right = tl.load($(candidate_vto_copy_116_ptr) + (((((((r335_group % $(8)) * $(8192)) + (r335_lane[:, None] * $(64)) + r335_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r335_group % $(8)) * $(8192)) + (r335_lane[:, None] * $(64)) + r335_col[None, :]) % $(64)) * $(128)) + (((((r335_group % $(8)) * $(8192)) + (r335_lane[:, None] * $(64)) + r335_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r335_value = tl.dot(r335_left, r335_right)
      r335_value = r335_value.to(tl.float32)
      tl.store($(candidate_vbmm_84_ptr) + ((r335_group * $(2048)) + (r335_row[:, None] * $(64)) + r335_col[None, :]), r335_value)
    }
    r336_z = tl.arange($(16384))
    r336_value = (tl.load($(candidate_vbmm_84_ptr) + ((((r336_z / $(2048)) % $(8)) * $(2048)) + (((r336_z / $(64)) % $(32)) * $(64)) + (r336_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_9_ptr) + ((((r336_z / $(2048)) % $(8)) * $(32)) + ((r336_z / $(64)) % $(32))), dtype=tl.float32))
    r336_value = r336_value.to(tl.float32)
    r336_value = tl.broadcast_to(r336_value, [$(16384)])
    tl.store($(candidate_vsub_43_ptr) + r336_z, r336_value)
    r337_z = tl.arange($(16384))
    r337_value = (tl.load($(candidate_vexp_4_ptr) + ((((r337_z / $(2048)) % $(8)) * $(2048)) + (((r337_z / $(64)) % $(32)) * $(64)) + (r337_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_43_ptr) + ((((r337_z / $(2048)) % $(8)) * $(2048)) + (((r337_z / $(64)) % $(32)) * $(64)) + (r337_z % $(64))), dtype=tl.float32))
    r337_value = r337_value.to(tl.float32)
    r337_value = tl.broadcast_to(r337_value, [$(16384)])
    tl.store($(candidate_vmul_57_ptr) + r337_z, r337_value)
    r338_row = tl.arange($(32))
    r338_col = tl.arange($(192))
    r338_lane = tl.arange($(64))
    for r338_group in range($(8)) {
      r338_left = tl.load($(candidate_vmul_57_ptr) + (((((((r338_group % $(8)) * $(2048)) + (r338_row[:, None] * $(64)) + r338_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r338_group % $(8)) * $(2048)) + (r338_row[:, None] * $(64)) + r338_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r338_group % $(8)) * $(2048)) + (r338_row[:, None] * $(64)) + r338_lane[None, :]) % $(64))), dtype=tl.float32)
      r338_right = tl.load($(candidate_vto_copy_115_ptr) + (((((((r338_group % $(8)) * $(12288)) + (r338_lane[:, None] * $(192)) + r338_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r338_group % $(8)) * $(12288)) + (r338_lane[:, None] * $(192)) + r338_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r338_group % $(8)) * $(12288)) + (r338_lane[:, None] * $(192)) + r338_col[None, :]) % $(192))), dtype=tl.float32)
      r338_value = tl.dot(r338_left, r338_right)
      r338_value = r338_value.to(tl.float32)
      tl.store($(candidate_vbmm_85_ptr) + ((r338_group * $(6144)) + (r338_row[:, None] * $(192)) + r338_col[None, :]), r338_value)
    }
    r339_z = tl.arange($(49152))
    r339_value = (tl.load($(candidate_vbmm_85_ptr) + ((((r339_z / $(6144)) % $(8)) * $(6144)) + (((r339_z / $(192)) % $(32)) * $(192)) + (r339_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r339_value = r339_value.to(tl.float32)
    r339_value = tl.broadcast_to(r339_value, [$(49152)])
    tl.store($(candidate_vmul_58_ptr) + r339_z, r339_value)
    r340_z = tl.arange($(49152))
    r340_value = tl.load($(candidate_vmul_58_ptr) + ((((r340_z / $(192)) % $(8)) * $(6144)) + (((r340_z / $(1536)) % $(32)) * $(192)) + (r340_z % $(192))), dtype=tl.float32)
    r340_value = r340_value.to(tl.bfloat16)
    r340_value = tl.broadcast_to(r340_value, [$(49152)])
    tl.store($(candidate_vto_copy_119_ptr) + r340_z, r340_value)
    r341_row = tl.arange($(64))
    r341_col = tl.arange($(192))
    r341_lane = tl.arange($(32))
    for r341_group in range($(8)) {
      r341_left = tl.load($(candidate_vmul_57_ptr) + (((((((r341_group % $(8)) * $(2048)) + (r341_row[:, None] * $(32)) + r341_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r341_group % $(8)) * $(2048)) + (r341_row[:, None] * $(32)) + r341_lane[None, :]) % $(32)) * $(64)) + (((((r341_group % $(8)) * $(2048)) + (r341_row[:, None] * $(32)) + r341_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r341_right = tl.load($(candidate_vto_copy_114_ptr) + (((((((r341_group % $(8)) * $(6144)) + (r341_lane[:, None] * $(192)) + r341_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r341_group % $(8)) * $(6144)) + (r341_lane[:, None] * $(192)) + r341_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r341_group % $(8)) * $(6144)) + (r341_lane[:, None] * $(192)) + r341_col[None, :]) % $(192))), dtype=tl.float32)
      r341_value = tl.dot(r341_left, r341_right)
      r341_value = r341_value.to(tl.float32)
      tl.store($(candidate_vbmm_86_ptr) + ((r341_group * $(12288)) + (r341_row[:, None] * $(192)) + r341_col[None, :]), r341_value)
    }
    r342_z = tl.arange($(98304))
    r342_value = (tl.load($(candidate_vbmm_86_ptr) + ((((r342_z / $(12288)) % $(8)) * $(12288)) + (((r342_z / $(192)) % $(64)) * $(192)) + (r342_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r342_value = r342_value.to(tl.float32)
    r342_value = tl.broadcast_to(r342_value, [$(98304)])
    tl.store($(candidate_vmul_59_ptr) + r342_z, r342_value)
    r343_z = tl.arange($(98304))
    r343_value = tl.load($(candidate_vmul_59_ptr) + ((((r343_z / $(192)) % $(8)) * $(12288)) + (((r343_z / $(1536)) % $(64)) * $(192)) + (r343_z % $(192))), dtype=tl.float32)
    r343_value = r343_value.to(tl.bfloat16)
    r343_value = tl.broadcast_to(r343_value, [$(98304)])
    tl.store($(candidate_vto_copy_120_ptr) + r343_z, r343_value)
    r344_row = tl.arange($(64))
    r344_col = tl.arange($(128))
    r344_lane = tl.arange($(32))
    for r344_group in range($(8)) {
      r344_left = tl.load($(candidate_vexp_4_ptr) + (((((((r344_group % $(8)) * $(2048)) + (r344_row[:, None] * $(32)) + r344_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r344_group % $(8)) * $(2048)) + (r344_row[:, None] * $(32)) + r344_lane[None, :]) % $(32)) * $(64)) + (((((r344_group % $(8)) * $(2048)) + (r344_row[:, None] * $(32)) + r344_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r344_right = tl.load($(candidate_vto_copy_117_ptr) + (((((((r344_group % $(8)) * $(4096)) + (r344_lane[:, None] * $(128)) + r344_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r344_group % $(8)) * $(4096)) + (r344_lane[:, None] * $(128)) + r344_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r344_group % $(8)) * $(4096)) + (r344_lane[:, None] * $(128)) + r344_col[None, :]) % $(128))), dtype=tl.float32)
      r344_value = tl.dot(r344_left, r344_right)
      r344_value = r344_value.to(tl.float32)
      tl.store($(candidate_vbmm_87_ptr) + ((r344_group * $(8192)) + (r344_row[:, None] * $(128)) + r344_col[None, :]), r344_value)
    }
    r345_z = tl.arange($(65536))
    r345_value = tl.load($(candidate_vbmm_87_ptr) + ((((r345_z / $(128)) % $(8)) * $(8192)) + (((r345_z / $(1024)) % $(64)) * $(128)) + (r345_z % $(128))), dtype=tl.float32)
    r345_value = r345_value.to(tl.bfloat16)
    r345_value = tl.broadcast_to(r345_value, [$(65536)])
    tl.store($(candidate_vto_copy_121_ptr) + r345_z, r345_value)
    r346_z = tl.arange($(147456))
    r346_value = tl.where(((r346_z / $(1536)) % $(96)) < $(64), tl.load($(candidate_vto_copy_120_ptr) + ((((((r346_z / $(1536)) % $(96)) - $(0)) % $(64)) * $(1536)) + (((r346_z / $(192)) % $(8)) * $(192)) + (r346_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r346_value = r346_value.to(tl.bfloat16)
    r346_value = tl.broadcast_to(r346_value, [$(147456)])
    tl.store($(candidate_vcat_58_ptr) + r346_z, r346_value)
    r347_z = tl.arange($(98304))
    r347_value = tl.where(((r347_z / $(1024)) % $(96)) < $(64), tl.load($(candidate_vto_copy_121_ptr) + ((((((r347_z / $(1024)) % $(96)) - $(0)) % $(64)) * $(1024)) + (((r347_z / $(128)) % $(8)) * $(128)) + (r347_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r347_value = r347_value.to(tl.bfloat16)
    r347_value = tl.broadcast_to(r347_value, [$(98304)])
    tl.store($(candidate_vcat_59_ptr) + r347_z, r347_value)
    r348_z = tl.arange($(49152))
    r348_value = tl.load($(candidate_vto_copy_119_ptr) + ((((((r348_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(1536)) + (((r348_z / $(192)) % $(8)) * $(192)) + (r348_z % $(192))), dtype=tl.bfloat16)
    r348_value = r348_value.to(tl.bfloat16)
    r348_value = tl.broadcast_to(r348_value, [$(49152)])
    tl.store($(candidate_vcat_60_ptr) + r348_z, r348_value)
    r349_z = tl.arange($(147456))
    r349_value = tl.load($(candidate_vcat_58_ptr) + ((((((r349_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r349_z / $(192)) % $(8)) * $(192)) + (r349_z % $(192))), dtype=tl.bfloat16)
    r349_value = r349_value.to(tl.bfloat16)
    r349_value = tl.broadcast_to(r349_value, [$(147456)])
    tl.store($(candidate_vcat_61_ptr) + r349_z, r349_value)
    r350_z = tl.arange($(98304))
    r350_value = tl.load($(candidate_vcat_59_ptr) + ((((((r350_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r350_z / $(128)) % $(8)) * $(128)) + (r350_z % $(128))), dtype=tl.bfloat16)
    r350_value = r350_value.to(tl.bfloat16)
    r350_value = tl.broadcast_to(r350_value, [$(98304)])
    tl.store($(candidate_vcat_62_ptr) + r350_z, r350_value)
    r351_z = tl.arange($(6144))
    r351_lane = tl.arange($(8))
    r351_value = tl.sum(tl.load($(candidate_vcat_61_ptr) + ((((r351_z[:, None] / $(64)) % $(96)) * $(1536)) + (r351_lane[None, :] * $(192)) + (r351_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r351_value = r351_value.to(tl.bfloat16)
    r351_value = tl.broadcast_to(r351_value, [$(6144)])
    tl.store($(candidate_vsum_10_ptr) + r351_z, r351_value)
    r352_row = tl.arange($(96))
    r352_col = tl.arange($(512))
    r352_lane = tl.arange($(1024))
    for r352_group in range($(1)) {
      r352_left = tl.load($(candidate_vcat_61_ptr) + ((((((r352_row[:, None] * $(1024)) + r352_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + ((((r352_row[:, None] * $(1024)) + r352_lane[None, :]) % $(8)) * $(192)) + ((((r352_row[:, None] * $(1024)) + r352_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r352_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r352_lane[:, None] * $(512)) + r352_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r352_lane[:, None] * $(512)) + r352_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r352_lane[:, None] * $(512)) + r352_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r352_value = tl.dot(r352_left, r352_right)
      r352_value = r352_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_88_ptr) + ((r352_group * $(49152)) + (r352_row[:, None] * $(512)) + r352_col[None, :]), r352_value)
    }
    r353_row = tl.arange($(96))
    r353_col = tl.arange($(512))
    r353_lane = tl.arange($(1024))
    for r353_group in range($(1)) {
      r353_left = tl.load($(candidate_vcat_62_ptr) + ((((((r353_row[:, None] * $(1024)) + r353_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + ((((r353_row[:, None] * $(1024)) + r353_lane[None, :]) % $(8)) * $(128)) + ((((r353_row[:, None] * $(1024)) + r353_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r353_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r353_lane[:, None] * $(512)) + r353_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r353_lane[:, None] * $(512)) + r353_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r353_lane[:, None] * $(512)) + r353_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r353_value = tl.dot(r353_left, r353_right)
      r353_value = r353_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_89_ptr) + ((r353_group * $(49152)) + (r353_row[:, None] * $(512)) + r353_col[None, :]), r353_value)
    }
    r354_z = tl.arange($(49152))
    r354_value = (tl.load($(candidate_vbmm_88_ptr) + ((((r354_z / $(512)) % $(96)) * $(512)) + (r354_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_89_ptr) + ((((r354_z / $(512)) % $(96)) * $(512)) + (r354_z % $(512))), dtype=tl.bfloat16))
    r354_value = r354_value.to(tl.bfloat16)
    r354_value = tl.broadcast_to(r354_value, [$(49152)])
    tl.store($(candidate_vadd_50_ptr) + r354_z, r354_value)
    r355_z = tl.arange($(147456))
    r355_value = tl.where(((((r355_z / $(512)) % $(288)) >= $(96)) & (((r355_z / $(512)) % $(288)) < $(192)) & ((((r355_z / $(512)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vadd_50_ptr) + ((((((r355_z / $(512)) % $(288)) - $(96)) % $(96)) * $(512)) + (r355_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r355_value = r355_value.to(tl.bfloat16)
    r355_value = tl.broadcast_to(r355_value, [$(147456)])
    tl.store($(candidate_vslice_backward_16_ptr) + r355_z, r355_value)
    r356_z = tl.arange($(147456))
    r356_value = (tl.load($(candidate_vadd_44_ptr) + ((((r356_z / $(512)) % $(288)) * $(512)) + (r356_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_16_ptr) + ((((r356_z / $(512)) % $(288)) * $(512)) + (r356_z % $(512))), dtype=tl.bfloat16))
    r356_value = r356_value.to(tl.bfloat16)
    r356_value = tl.broadcast_to(r356_value, [$(147456)])
    tl.store($(candidate_vadd_51_ptr) + r356_z, r356_value)
    r357_z = tl.arange($(18432))
    r357_value = tl.where(((((r357_z / $(64)) % $(288)) >= $(96)) & (((r357_z / $(64)) % $(288)) < $(192)) & ((((r357_z / $(64)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vsum_10_ptr) + ((((((r357_z / $(64)) % $(288)) - $(96)) % $(96)) * $(64)) + (r357_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r357_value = r357_value.to(tl.bfloat16)
    r357_value = tl.broadcast_to(r357_value, [$(18432)])
    tl.store($(candidate_vslice_backward_17_ptr) + r357_z, r357_value)
    r358_z = tl.arange($(18432))
    r358_value = (tl.load($(candidate_vadd_45_ptr) + ((((r358_z / $(64)) % $(288)) * $(64)) + (r358_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_17_ptr) + ((((r358_z / $(64)) % $(288)) * $(64)) + (r358_z % $(64))), dtype=tl.bfloat16))
    r358_value = r358_value.to(tl.bfloat16)
    r358_value = tl.broadcast_to(r358_value, [$(18432)])
    tl.store($(candidate_vadd_52_ptr) + r358_z, r358_value)
    r359_z = tl.arange($(294912))
    r359_value = tl.where(((((r359_z / $(1024)) % $(288)) >= $(128)) & (((r359_z / $(1024)) % $(288)) < $(160)) & ((((r359_z / $(1024)) % $(288)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vcat_60_ptr) + ((((((r359_z / $(1024)) % $(288)) - $(128)) % $(32)) * $(1536)) + (((r359_z / $(128)) % $(8)) * $(192)) + (r359_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r359_value = r359_value.to(tl.bfloat16)
    r359_value = tl.broadcast_to(r359_value, [$(294912)])
    tl.store($(candidate_vslice_backward_18_ptr) + r359_z, r359_value)
    r360_z = tl.arange($(294912))
    r360_value = (tl.load($(candidate_vadd_46_ptr) + ((((r360_z / $(1024)) % $(288)) * $(1024)) + (((r360_z / $(128)) % $(8)) * $(128)) + (r360_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_18_ptr) + ((((r360_z / $(1024)) % $(288)) * $(1024)) + (((r360_z / $(128)) % $(8)) * $(128)) + (r360_z % $(128))), dtype=tl.bfloat16))
    r360_value = r360_value.to(tl.bfloat16)
    r360_value = tl.broadcast_to(r360_value, [$(294912)])
    tl.store($(candidate_vadd_53_ptr) + r360_z, r360_value)
    r361_z = tl.arange($(147456))
    r361_value = tl.where(((((r361_z / $(512)) % $(288)) >= $(128)) & (((r361_z / $(512)) % $(288)) < $(160)) & ((((r361_z / $(512)) % $(288)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vcat_60_ptr) + ((((((r361_z / $(512)) % $(288)) - $(128)) % $(32)) * $(1536)) + (((r361_z / $(64)) % $(8)) * $(192)) + (r361_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r361_value = r361_value.to(tl.bfloat16)
    r361_value = tl.broadcast_to(r361_value, [$(147456)])
    tl.store($(candidate_vslice_backward_19_ptr) + r361_z, r361_value)
    r362_z = tl.arange($(147456))
    r362_value = (tl.load($(candidate_vadd_47_ptr) + ((((r362_z / $(512)) % $(288)) * $(512)) + (((r362_z / $(64)) % $(8)) * $(64)) + (r362_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_19_ptr) + ((((r362_z / $(512)) % $(288)) * $(512)) + (((r362_z / $(64)) % $(8)) * $(64)) + (r362_z % $(64))), dtype=tl.bfloat16))
    r362_value = r362_value.to(tl.bfloat16)
    r362_value = tl.broadcast_to(r362_value, [$(147456)])
    tl.store($(candidate_vadd_54_ptr) + r362_z, r362_value)
    r363_row = tl.arange($(96))
    r363_col = tl.arange($(1024))
    r363_lane = tl.arange($(512))
    for r363_group in range($(1)) {
      r363_left = tl.load($(candidate_vcat_7_ptr) + ((((((r363_row[:, None] * $(512)) + r363_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r363_row[:, None] * $(512)) + r363_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r363_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r363_lane[:, None] * $(1024)) + r363_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r363_lane[:, None] * $(1024)) + r363_col[None, :]) % $(128)) * $(512)) + ((((r363_lane[:, None] * $(1024)) + r363_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r363_value = tl.dot(r363_left, r363_right)
      r363_value = r363_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_92_ptr) + ((r363_group * $(98304)) + (r363_row[:, None] * $(1024)) + r363_col[None, :]), r363_value)
    }
    r364_row = tl.arange($(96))
    r364_col = tl.arange($(1024))
    r364_lane = tl.arange($(512))
    for r364_group in range($(1)) {
      r364_left = tl.load($(candidate_vcat_7_ptr) + ((((((r364_row[:, None] * $(512)) + r364_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r364_row[:, None] * $(512)) + r364_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r364_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r364_lane[:, None] * $(1024)) + r364_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r364_lane[:, None] * $(1024)) + r364_col[None, :]) % $(128)) * $(512)) + ((((r364_lane[:, None] * $(1024)) + r364_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r364_value = tl.dot(r364_left, r364_right)
      r364_value = r364_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_93_ptr) + ((r364_group * $(98304)) + (r364_row[:, None] * $(1024)) + r364_col[None, :]), r364_value)
    }
    r365_z = tl.arange($(49152))
    r365_value = tl.where((r365_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r365_z / $(1536)) % $(32)) * $(1024)) + (((r365_z / $(192)) % $(8)) * $(128)) + ((r365_z % $(192)) - $(0)) + $(98304)) / $(1024)) % $(288)) * $(1536)) + (((((((r365_z / $(1536)) % $(32)) * $(1024)) + (((r365_z / $(192)) % $(8)) * $(128)) + ((r365_z % $(192)) - $(0)) + $(98304)) / $(128)) % $(8)) * $(192)) + (((((r365_z / $(1536)) % $(32)) * $(1024)) + (((r365_z / $(192)) % $(8)) * $(128)) + ((r365_z % $(192)) - $(0)) + $(98304)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r365_z / $(1536)) % $(32)) * $(512)) + (((r365_z / $(192)) % $(8)) * $(64)) + ((r365_z % $(192)) - $(128)) + $(49152)) / $(64)) % $(8)) * $(18432)) + (((((((r365_z / $(1536)) % $(32)) * $(512)) + (((r365_z / $(192)) % $(8)) * $(64)) + ((r365_z % $(192)) - $(128)) + $(49152)) / $(512)) % $(288)) * $(64)) + (((((r365_z / $(1536)) % $(32)) * $(512)) + (((r365_z / $(192)) % $(8)) * $(64)) + ((r365_z % $(192)) - $(128)) + $(49152)) % $(64))), dtype=tl.bfloat16))
    r365_value = r365_value.to(tl.bfloat16)
    r365_value = tl.broadcast_to(r365_value, [$(49152)])
    tl.store($(candidate_vcat_64_ptr) + r365_z, r365_value)
    r366_z = tl.arange($(147456))
    r366_value = tl.where((r366_z % $(192)) < $(128), tl.load($(candidate_vbmm_92_ptr) + ((((((((r366_z / $(1536)) % $(96)) * $(1024)) + (((r366_z / $(192)) % $(8)) * $(128)) + ((r366_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r366_z / $(1536)) % $(96)) * $(1024)) + (((r366_z / $(192)) % $(8)) * $(128)) + ((r366_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r366_z / $(1536)) % $(96)) * $(1024)) + (((r366_z / $(192)) % $(8)) * $(128)) + ((r366_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r366_z / $(1536)) % $(96)) * $(576)) + ((((r366_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r366_value = r366_value.to(tl.bfloat16)
    r366_value = tl.broadcast_to(r366_value, [$(147456)])
    tl.store($(candidate_vcat_65_ptr) + r366_z, r366_value)
    r367_z = tl.arange($(49152))
    r367_value = tl.load($(candidate_vcat_64_ptr) + ((((r367_z / $(192)) % $(32)) * $(1536)) + (((r367_z / $(6144)) % $(8)) * $(192)) + (r367_z % $(192))), dtype=tl.bfloat16)
    r367_value = r367_value.to(tl.float32)
    r367_value = tl.broadcast_to(r367_value, [$(49152)])
    tl.store($(candidate_vto_copy_122_ptr) + r367_z, r367_value)
    r368_z = tl.arange($(49152))
    r368_value = tl.load($(candidate_vcat_65_ptr) + ((((r368_z / $(192)) % $(32)) * $(1536)) + (((r368_z / $(6144)) % $(8)) * $(192)) + (r368_z % $(192))), dtype=tl.bfloat16)
    r368_value = r368_value.to(tl.float32)
    r368_value = tl.broadcast_to(r368_value, [$(49152)])
    tl.store($(candidate_vto_copy_123_ptr) + r368_z, r368_value)
    r369_row = tl.arange($(32))
    r369_col = tl.arange($(32))
    r369_lane = tl.arange($(192))
    for r369_group in range($(8)) {
      r369_left = tl.load($(candidate_vto_copy_122_ptr) + (((((((r369_group % $(8)) * $(6144)) + (r369_row[:, None] * $(192)) + r369_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r369_group % $(8)) * $(6144)) + (r369_row[:, None] * $(192)) + r369_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r369_group % $(8)) * $(6144)) + (r369_row[:, None] * $(192)) + r369_lane[None, :]) % $(192))), dtype=tl.float32)
      r369_right = tl.load($(candidate_vto_copy_123_ptr) + (((((((r369_group % $(8)) * $(6144)) + (r369_lane[:, None] * $(32)) + r369_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r369_group % $(8)) * $(6144)) + (r369_lane[:, None] * $(32)) + r369_col[None, :]) % $(32)) * $(192)) + (((((r369_group % $(8)) * $(6144)) + (r369_lane[:, None] * $(32)) + r369_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r369_value = tl.dot(r369_left, r369_right)
      r369_value = r369_value.to(tl.float32)
      tl.store($(candidate_vbmm_94_ptr) + ((r369_group * $(1024)) + (r369_row[:, None] * $(32)) + r369_col[None, :]), r369_value)
    }
    r370_z = tl.arange($(8192))
    r370_value = (tl.load($(candidate_vbmm_94_ptr) + ((((r370_z / $(1024)) % $(8)) * $(1024)) + (((r370_z / $(32)) % $(32)) * $(32)) + (r370_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r370_value = r370_value.to(tl.float32)
    r370_value = tl.broadcast_to(r370_value, [$(8192)])
    tl.store($(candidate_vmul_60_ptr) + r370_z, r370_value)
    r371_z = tl.arange($(32))
    r371_value = (tl.cast(r371_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r371_value = tl.broadcast_to(r371_value, [$(32)])
    tl.store($(candidate_vadd_55_ptr) + r371_z, r371_value)
    r372_z = tl.arange($(32))
    r372_value = (tl.load($(candidate_vadd_55_ptr) + r372_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r372_value = tl.broadcast_to(r372_value, [$(32)])
    tl.store($(candidate_vsub_44_ptr) + r372_z, r372_value)
    r373_z = tl.arange($(1024))
    r373_value = (tl.cast((r373_z % $(32)), tl.int64) > tl.load($(candidate_vsub_44_ptr) + ((r373_z / $(32)) % $(32)), dtype=tl.int64))
    r373_value = tl.broadcast_to(r373_value, [$(1024)])
    tl.store($(candidate_vgt_19_ptr) + r373_z, r373_value)
    r374_z = tl.arange($(8192))
    r374_value = tl.where(tl.load($(candidate_vgt_19_ptr) + ((((r374_z / $(32)) % $(32)) * $(32)) + (r374_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_60_ptr) + ((((r374_z / $(1024)) % $(8)) * $(1024)) + (((r374_z / $(32)) % $(32)) * $(32)) + (r374_z % $(32))), dtype=tl.float32))
    r374_value = r374_value.to(tl.float32)
    r374_value = tl.broadcast_to(r374_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_19_ptr) + r374_z, r374_value)
    r375_z = tl.arange($(8192))
    r375_lane = tl.arange($(32))
    r375_row_values = tl.load($(candidate_vmasked_fill_19_ptr) + ((((r375_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r375_z[:, None] / $(32)) % $(32)) * $(32)) + (r375_lane[None, :] % $(32))), dtype=tl.float32)
    r375_maximum = tl.max(r375_row_values, axis=1, keep_dims=true)
    r375_weights = tl.exp(r375_row_values - r375_maximum)
    r375_total = tl.sum(r375_weights, axis=1)
    r375_maximum = tl.max(r375_row_values, axis=1)
    r375_value = tl.exp(tl.load($(candidate_vmasked_fill_19_ptr) + ((((r375_z / $(1024)) % $(8)) * $(1024)) + (((r375_z / $(32)) % $(32)) * $(32)) + (r375_z % $(32))), dtype=tl.float32) - r375_maximum) / r375_total
    r375_value = r375_value.to(tl.float32)
    r375_value = tl.broadcast_to(r375_value, [$(8192)])
    tl.store($(candidate_vsoftmax_14_ptr) + r375_z, r375_value)
    r376_z = tl.arange($(8192))
    r376_value = tl.load($(candidate_vsoftmax_14_ptr) + ((((r376_z / $(1024)) % $(8)) * $(1024)) + (((r376_z / $(32)) % $(32)) * $(32)) + (r376_z % $(32))), dtype=tl.float32)
    r376_value = r376_value.to(tl.bfloat16)
    r376_value = tl.broadcast_to(r376_value, [$(8192)])
    tl.store($(candidate_vto_copy_124_ptr) + r376_z, r376_value)
    r377_row = tl.arange($(32))
    r377_col = tl.arange($(128))
    r377_lane = tl.arange($(32))
    for r377_group in range($(8)) {
      r377_left = tl.load($(candidate_vto_copy_124_ptr) + (((((((r377_group % $(8)) * $(1024)) + (r377_row[:, None] * $(32)) + r377_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r377_group % $(8)) * $(1024)) + (r377_row[:, None] * $(32)) + r377_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r377_group % $(8)) * $(1024)) + (r377_row[:, None] * $(32)) + r377_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r377_right = tl.load($(candidate_vbmm_93_ptr) + (((((((r377_group % $(8)) * $(4096)) + (r377_lane[:, None] * $(128)) + r377_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r377_group % $(8)) * $(4096)) + (r377_lane[:, None] * $(128)) + r377_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r377_group % $(8)) * $(4096)) + (r377_lane[:, None] * $(128)) + r377_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r377_value = tl.dot(r377_left, r377_right)
      r377_value = r377_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_95_ptr) + ((r377_group * $(4096)) + (r377_row[:, None] * $(128)) + r377_col[None, :]), r377_value)
    }
    r378_z = tl.arange($(256))
    r378_lane = tl.arange($(32))
    r378_row_values = tl.load($(candidate_vmasked_fill_19_ptr) + ((((r378_z[:, None] / $(32)) % $(8)) * $(1024)) + ((r378_z[:, None] % $(32)) * $(32)) + (r378_lane[None, :] % $(32))), dtype=tl.float32)
    r378_maximum = tl.max(r378_row_values, axis=1, keep_dims=true)
    r378_weights = tl.exp(r378_row_values - r378_maximum)
    r378_total = tl.sum(r378_weights, axis=1)
    r378_maximum = tl.max(r378_row_values, axis=1)
    r378_value = r378_maximum + tl.log(r378_total)
    r378_value = r378_value.to(tl.float32)
    r378_value = tl.broadcast_to(r378_value, [$(256)])
    tl.store($(candidate_vlogsumexp_14_ptr) + r378_z, r378_value)
    r379_z = tl.arange($(32768))
    r379_value = tl.load($(candidate_vbmm_95_ptr) + ((((((((r379_z / $(128)) % $(8)) * $(4096)) + ((((r379_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r379_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r379_z / $(128)) % $(8)) * $(32)) + (((r379_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r379_z % $(128))), dtype=tl.bfloat16)
    r379_value = r379_value.to(tl.bfloat16)
    r379_value = tl.broadcast_to(r379_value, [$(32768)])
    tl.store($(candidate_vcat_66_ptr) + r379_z, r379_value)
    r380_z = tl.arange($(256))
    r380_value = tl.load($(candidate_vlogsumexp_14_ptr) + (((r380_z % $(8)) * $(32)) + ((((r380_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r380_value = r380_value.to(tl.float32)
    r380_value = tl.broadcast_to(r380_value, [$(256)])
    tl.store($(candidate_vcat_67_ptr) + r380_z, r380_value)
    r381_z = tl.arange($(49152))
    r381_value = tl.load($(candidate_vcat_64_ptr) + ((((r381_z / $(192)) % $(32)) * $(1536)) + (((r381_z / $(6144)) % $(8)) * $(192)) + (r381_z % $(192))), dtype=tl.bfloat16)
    r381_value = r381_value.to(tl.float32)
    r381_value = tl.broadcast_to(r381_value, [$(49152)])
    tl.store($(candidate_vto_copy_125_ptr) + r381_z, r381_value)
    r382_z = tl.arange($(49152))
    r382_value = tl.load($(candidate_vcat_65_ptr) + ((((r382_z / $(192)) % $(32)) * $(1536)) + (((r382_z / $(6144)) % $(8)) * $(192)) + (r382_z % $(192))), dtype=tl.bfloat16)
    r382_value = r382_value.to(tl.float32)
    r382_value = tl.broadcast_to(r382_value, [$(49152)])
    tl.store($(candidate_vto_copy_126_ptr) + r382_z, r382_value)
    r383_z = tl.arange($(32768))
    r383_value = tl.load($(candidate_vbmm_93_ptr) + ((((r383_z / $(128)) % $(32)) * $(1024)) + (((r383_z / $(4096)) % $(8)) * $(128)) + (r383_z % $(128))), dtype=tl.bfloat16)
    r383_value = r383_value.to(tl.float32)
    r383_value = tl.broadcast_to(r383_value, [$(32768)])
    tl.store($(candidate_vto_copy_127_ptr) + r383_z, r383_value)
    r384_z = tl.arange($(32768))
    r384_value = tl.load($(candidate_vmm_22_ptr) + ((((r384_z / $(128)) % $(32)) * $(1024)) + (((r384_z / $(4096)) % $(8)) * $(128)) + (r384_z % $(128)) + $(98304)), dtype=tl.bfloat16)
    r384_value = r384_value.to(tl.float32)
    r384_value = tl.broadcast_to(r384_value, [$(32768)])
    tl.store($(candidate_vto_copy_128_ptr) + r384_z, r384_value)
    r385_z = tl.arange($(32768))
    r385_value = tl.load($(candidate_vcat_66_ptr) + ((((r385_z / $(128)) % $(32)) * $(1024)) + (((r385_z / $(4096)) % $(8)) * $(128)) + (r385_z % $(128))), dtype=tl.bfloat16)
    r385_value = r385_value.to(tl.float32)
    r385_value = tl.broadcast_to(r385_value, [$(32768)])
    tl.store($(candidate_vto_copy_129_ptr) + r385_z, r385_value)
    r386_row = tl.arange($(32))
    r386_col = tl.arange($(32))
    r386_lane = tl.arange($(192))
    for r386_group in range($(8)) {
      r386_left = tl.load($(candidate_vto_copy_125_ptr) + (((((((r386_group % $(8)) * $(6144)) + (r386_row[:, None] * $(192)) + r386_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r386_group % $(8)) * $(6144)) + (r386_row[:, None] * $(192)) + r386_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r386_group % $(8)) * $(6144)) + (r386_row[:, None] * $(192)) + r386_lane[None, :]) % $(192))), dtype=tl.float32)
      r386_right = tl.load($(candidate_vto_copy_126_ptr) + (((((((r386_group % $(8)) * $(6144)) + (r386_lane[:, None] * $(32)) + r386_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r386_group % $(8)) * $(6144)) + (r386_lane[:, None] * $(32)) + r386_col[None, :]) % $(32)) * $(192)) + (((((r386_group % $(8)) * $(6144)) + (r386_lane[:, None] * $(32)) + r386_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r386_value = tl.dot(r386_left, r386_right)
      r386_value = r386_value.to(tl.float32)
      tl.store($(candidate_vbmm_96_ptr) + ((r386_group * $(1024)) + (r386_row[:, None] * $(32)) + r386_col[None, :]), r386_value)
    }
    r387_z = tl.arange($(8192))
    r387_value = (tl.load($(candidate_vbmm_96_ptr) + ((((r387_z / $(1024)) % $(8)) * $(1024)) + (((r387_z / $(32)) % $(32)) * $(32)) + (r387_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r387_value = r387_value.to(tl.float32)
    r387_value = tl.broadcast_to(r387_value, [$(8192)])
    tl.store($(candidate_vmul_61_ptr) + r387_z, r387_value)
    r388_z = tl.arange($(32))
    r388_value = (tl.cast(r388_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r388_value = tl.broadcast_to(r388_value, [$(32)])
    tl.store($(candidate_vadd_56_ptr) + r388_z, r388_value)
    r389_z = tl.arange($(32))
    r389_value = (tl.load($(candidate_vadd_56_ptr) + r389_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r389_value = tl.broadcast_to(r389_value, [$(32)])
    tl.store($(candidate_vsub_45_ptr) + r389_z, r389_value)
    r390_z = tl.arange($(1024))
    r390_value = (tl.cast((r390_z % $(32)), tl.int64) > tl.load($(candidate_vsub_45_ptr) + ((r390_z / $(32)) % $(32)), dtype=tl.int64))
    r390_value = tl.broadcast_to(r390_value, [$(1024)])
    tl.store($(candidate_vgt_20_ptr) + r390_z, r390_value)
    r391_z = tl.arange($(8192))
    r391_value = tl.where(tl.load($(candidate_vgt_20_ptr) + ((((r391_z / $(32)) % $(32)) * $(32)) + (r391_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_61_ptr) + ((((r391_z / $(1024)) % $(8)) * $(1024)) + (((r391_z / $(32)) % $(32)) * $(32)) + (r391_z % $(32))), dtype=tl.float32))
    r391_value = r391_value.to(tl.float32)
    r391_value = tl.broadcast_to(r391_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_20_ptr) + r391_z, r391_value)
    r392_z = tl.arange($(8192))
    r392_value = (tl.load($(candidate_vmasked_fill_20_ptr) + ((((r392_z / $(1024)) % $(8)) * $(1024)) + (((r392_z / $(32)) % $(32)) * $(32)) + (r392_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_67_ptr) + ((((r392_z / $(32)) % $(32)) * $(8)) + ((r392_z / $(1024)) % $(8))), dtype=tl.float32))
    r392_value = r392_value.to(tl.float32)
    r392_value = tl.broadcast_to(r392_value, [$(8192)])
    tl.store($(candidate_vsub_46_ptr) + r392_z, r392_value)
    r393_z = tl.arange($(8192))
    r393_value = tl.exp(tl.load($(candidate_vsub_46_ptr) + ((((r393_z / $(1024)) % $(8)) * $(1024)) + (((r393_z / $(32)) % $(32)) * $(32)) + (r393_z % $(32))), dtype=tl.float32))
    r393_value = r393_value.to(tl.float32)
    r393_value = tl.broadcast_to(r393_value, [$(8192)])
    tl.store($(candidate_vexp_5_ptr) + r393_z, r393_value)
    r394_z = tl.arange($(32768))
    r394_value = (tl.load($(candidate_vto_copy_128_ptr) + ((((r394_z / $(4096)) % $(8)) * $(4096)) + (((r394_z / $(128)) % $(32)) * $(128)) + (r394_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_129_ptr) + ((((r394_z / $(4096)) % $(8)) * $(4096)) + (((r394_z / $(128)) % $(32)) * $(128)) + (r394_z % $(128))), dtype=tl.float32))
    r394_value = r394_value.to(tl.float32)
    r394_value = tl.broadcast_to(r394_value, [$(32768)])
    tl.store($(candidate_vmul_62_ptr) + r394_z, r394_value)
    r395_z = tl.arange($(256))
    r395_lane = tl.arange($(128))
    r395_value = tl.sum(tl.load($(candidate_vmul_62_ptr) + ((((r395_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r395_z[:, None] % $(32)) * $(128)) + r395_lane[None, :]), dtype=tl.float32), axis=1)
    r395_value = r395_value.to(tl.float32)
    r395_value = tl.broadcast_to(r395_value, [$(256)])
    tl.store($(candidate_vsum_11_ptr) + r395_z, r395_value)
    r396_row = tl.arange($(32))
    r396_col = tl.arange($(32))
    r396_lane = tl.arange($(128))
    for r396_group in range($(8)) {
      r396_left = tl.load($(candidate_vto_copy_128_ptr) + (((((((r396_group % $(8)) * $(4096)) + (r396_row[:, None] * $(128)) + r396_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r396_group % $(8)) * $(4096)) + (r396_row[:, None] * $(128)) + r396_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r396_group % $(8)) * $(4096)) + (r396_row[:, None] * $(128)) + r396_lane[None, :]) % $(128))), dtype=tl.float32)
      r396_right = tl.load($(candidate_vto_copy_127_ptr) + (((((((r396_group % $(8)) * $(4096)) + (r396_lane[:, None] * $(32)) + r396_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r396_group % $(8)) * $(4096)) + (r396_lane[:, None] * $(32)) + r396_col[None, :]) % $(32)) * $(128)) + (((((r396_group % $(8)) * $(4096)) + (r396_lane[:, None] * $(32)) + r396_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r396_value = tl.dot(r396_left, r396_right)
      r396_value = r396_value.to(tl.float32)
      tl.store($(candidate_vbmm_97_ptr) + ((r396_group * $(1024)) + (r396_row[:, None] * $(32)) + r396_col[None, :]), r396_value)
    }
    r397_z = tl.arange($(8192))
    r397_value = (tl.load($(candidate_vbmm_97_ptr) + ((((r397_z / $(1024)) % $(8)) * $(1024)) + (((r397_z / $(32)) % $(32)) * $(32)) + (r397_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_11_ptr) + ((((r397_z / $(1024)) % $(8)) * $(32)) + ((r397_z / $(32)) % $(32))), dtype=tl.float32))
    r397_value = r397_value.to(tl.float32)
    r397_value = tl.broadcast_to(r397_value, [$(8192)])
    tl.store($(candidate_vsub_47_ptr) + r397_z, r397_value)
    r398_z = tl.arange($(8192))
    r398_value = (tl.load($(candidate_vexp_5_ptr) + ((((r398_z / $(1024)) % $(8)) * $(1024)) + (((r398_z / $(32)) % $(32)) * $(32)) + (r398_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_47_ptr) + ((((r398_z / $(1024)) % $(8)) * $(1024)) + (((r398_z / $(32)) % $(32)) * $(32)) + (r398_z % $(32))), dtype=tl.float32))
    r398_value = r398_value.to(tl.float32)
    r398_value = tl.broadcast_to(r398_value, [$(8192)])
    tl.store($(candidate_vmul_63_ptr) + r398_z, r398_value)
    r399_row = tl.arange($(32))
    r399_col = tl.arange($(192))
    r399_lane = tl.arange($(32))
    for r399_group in range($(8)) {
      r399_left = tl.load($(candidate_vmul_63_ptr) + (((((((r399_group % $(8)) * $(1024)) + (r399_row[:, None] * $(32)) + r399_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r399_group % $(8)) * $(1024)) + (r399_row[:, None] * $(32)) + r399_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r399_group % $(8)) * $(1024)) + (r399_row[:, None] * $(32)) + r399_lane[None, :]) % $(32))), dtype=tl.float32)
      r399_right = tl.load($(candidate_vto_copy_126_ptr) + (((((((r399_group % $(8)) * $(6144)) + (r399_lane[:, None] * $(192)) + r399_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r399_group % $(8)) * $(6144)) + (r399_lane[:, None] * $(192)) + r399_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r399_group % $(8)) * $(6144)) + (r399_lane[:, None] * $(192)) + r399_col[None, :]) % $(192))), dtype=tl.float32)
      r399_value = tl.dot(r399_left, r399_right)
      r399_value = r399_value.to(tl.float32)
      tl.store($(candidate_vbmm_98_ptr) + ((r399_group * $(6144)) + (r399_row[:, None] * $(192)) + r399_col[None, :]), r399_value)
    }
    r400_z = tl.arange($(49152))
    r400_value = (tl.load($(candidate_vbmm_98_ptr) + ((((r400_z / $(6144)) % $(8)) * $(6144)) + (((r400_z / $(192)) % $(32)) * $(192)) + (r400_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r400_value = r400_value.to(tl.float32)
    r400_value = tl.broadcast_to(r400_value, [$(49152)])
    tl.store($(candidate_vmul_64_ptr) + r400_z, r400_value)
    r401_z = tl.arange($(49152))
    r401_value = tl.load($(candidate_vmul_64_ptr) + ((((r401_z / $(192)) % $(8)) * $(6144)) + (((r401_z / $(1536)) % $(32)) * $(192)) + (r401_z % $(192))), dtype=tl.float32)
    r401_value = r401_value.to(tl.bfloat16)
    r401_value = tl.broadcast_to(r401_value, [$(49152)])
    tl.store($(candidate_vto_copy_130_ptr) + r401_z, r401_value)
    r402_row = tl.arange($(32))
    r402_col = tl.arange($(192))
    r402_lane = tl.arange($(32))
    for r402_group in range($(8)) {
      r402_left = tl.load($(candidate_vmul_63_ptr) + (((((((r402_group % $(8)) * $(1024)) + (r402_row[:, None] * $(32)) + r402_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r402_group % $(8)) * $(1024)) + (r402_row[:, None] * $(32)) + r402_lane[None, :]) % $(32)) * $(32)) + (((((r402_group % $(8)) * $(1024)) + (r402_row[:, None] * $(32)) + r402_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r402_right = tl.load($(candidate_vto_copy_125_ptr) + (((((((r402_group % $(8)) * $(6144)) + (r402_lane[:, None] * $(192)) + r402_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r402_group % $(8)) * $(6144)) + (r402_lane[:, None] * $(192)) + r402_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r402_group % $(8)) * $(6144)) + (r402_lane[:, None] * $(192)) + r402_col[None, :]) % $(192))), dtype=tl.float32)
      r402_value = tl.dot(r402_left, r402_right)
      r402_value = r402_value.to(tl.float32)
      tl.store($(candidate_vbmm_99_ptr) + ((r402_group * $(6144)) + (r402_row[:, None] * $(192)) + r402_col[None, :]), r402_value)
    }
    r403_z = tl.arange($(49152))
    r403_value = (tl.load($(candidate_vbmm_99_ptr) + ((((r403_z / $(6144)) % $(8)) * $(6144)) + (((r403_z / $(192)) % $(32)) * $(192)) + (r403_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r403_value = r403_value.to(tl.float32)
    r403_value = tl.broadcast_to(r403_value, [$(49152)])
    tl.store($(candidate_vmul_65_ptr) + r403_z, r403_value)
    r404_z = tl.arange($(49152))
    r404_value = tl.load($(candidate_vmul_65_ptr) + ((((r404_z / $(192)) % $(8)) * $(6144)) + (((r404_z / $(1536)) % $(32)) * $(192)) + (r404_z % $(192))), dtype=tl.float32)
    r404_value = r404_value.to(tl.bfloat16)
    r404_value = tl.broadcast_to(r404_value, [$(49152)])
    tl.store($(candidate_vto_copy_131_ptr) + r404_z, r404_value)
    r405_row = tl.arange($(32))
    r405_col = tl.arange($(128))
    r405_lane = tl.arange($(32))
    for r405_group in range($(8)) {
      r405_left = tl.load($(candidate_vexp_5_ptr) + (((((((r405_group % $(8)) * $(1024)) + (r405_row[:, None] * $(32)) + r405_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r405_group % $(8)) * $(1024)) + (r405_row[:, None] * $(32)) + r405_lane[None, :]) % $(32)) * $(32)) + (((((r405_group % $(8)) * $(1024)) + (r405_row[:, None] * $(32)) + r405_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r405_right = tl.load($(candidate_vto_copy_128_ptr) + (((((((r405_group % $(8)) * $(4096)) + (r405_lane[:, None] * $(128)) + r405_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r405_group % $(8)) * $(4096)) + (r405_lane[:, None] * $(128)) + r405_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r405_group % $(8)) * $(4096)) + (r405_lane[:, None] * $(128)) + r405_col[None, :]) % $(128))), dtype=tl.float32)
      r405_value = tl.dot(r405_left, r405_right)
      r405_value = r405_value.to(tl.float32)
      tl.store($(candidate_vbmm_100_ptr) + ((r405_group * $(4096)) + (r405_row[:, None] * $(128)) + r405_col[None, :]), r405_value)
    }
    r406_z = tl.arange($(32768))
    r406_value = tl.load($(candidate_vbmm_100_ptr) + ((((r406_z / $(128)) % $(8)) * $(4096)) + (((r406_z / $(1024)) % $(32)) * $(128)) + (r406_z % $(128))), dtype=tl.float32)
    r406_value = r406_value.to(tl.bfloat16)
    r406_value = tl.broadcast_to(r406_value, [$(32768)])
    tl.store($(candidate_vto_copy_132_ptr) + r406_z, r406_value)
    r407_z = tl.arange($(147456))
    r407_value = tl.where(((r407_z / $(1536)) % $(96)) < $(32), tl.load($(candidate_vto_copy_131_ptr) + ((((((r407_z / $(1536)) % $(96)) - $(0)) % $(32)) * $(1536)) + (((r407_z / $(192)) % $(8)) * $(192)) + (r407_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r407_value = r407_value.to(tl.bfloat16)
    r407_value = tl.broadcast_to(r407_value, [$(147456)])
    tl.store($(candidate_vcat_68_ptr) + r407_z, r407_value)
    r408_z = tl.arange($(98304))
    r408_value = tl.where(((r408_z / $(1024)) % $(96)) < $(32), tl.load($(candidate_vto_copy_132_ptr) + ((((((r408_z / $(1024)) % $(96)) - $(0)) % $(32)) * $(1024)) + (((r408_z / $(128)) % $(8)) * $(128)) + (r408_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r408_value = r408_value.to(tl.bfloat16)
    r408_value = tl.broadcast_to(r408_value, [$(98304)])
    tl.store($(candidate_vcat_69_ptr) + r408_z, r408_value)
    r409_z = tl.arange($(49152))
    r409_value = tl.load($(candidate_vto_copy_130_ptr) + ((((((r409_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(1536)) + (((r409_z / $(192)) % $(8)) * $(192)) + (r409_z % $(192))), dtype=tl.bfloat16)
    r409_value = r409_value.to(tl.bfloat16)
    r409_value = tl.broadcast_to(r409_value, [$(49152)])
    tl.store($(candidate_vcat_70_ptr) + r409_z, r409_value)
    r410_z = tl.arange($(147456))
    r410_value = tl.load($(candidate_vcat_68_ptr) + ((((((r410_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r410_z / $(192)) % $(8)) * $(192)) + (r410_z % $(192))), dtype=tl.bfloat16)
    r410_value = r410_value.to(tl.bfloat16)
    r410_value = tl.broadcast_to(r410_value, [$(147456)])
    tl.store($(candidate_vcat_71_ptr) + r410_z, r410_value)
    r411_z = tl.arange($(98304))
    r411_value = tl.load($(candidate_vcat_69_ptr) + ((((((r411_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r411_z / $(128)) % $(8)) * $(128)) + (r411_z % $(128))), dtype=tl.bfloat16)
    r411_value = r411_value.to(tl.bfloat16)
    r411_value = tl.broadcast_to(r411_value, [$(98304)])
    tl.store($(candidate_vcat_72_ptr) + r411_z, r411_value)
    r412_z = tl.arange($(6144))
    r412_lane = tl.arange($(8))
    r412_value = tl.sum(tl.load($(candidate_vcat_71_ptr) + ((((r412_z[:, None] / $(64)) % $(96)) * $(1536)) + (r412_lane[None, :] * $(192)) + (r412_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r412_value = r412_value.to(tl.bfloat16)
    r412_value = tl.broadcast_to(r412_value, [$(6144)])
    tl.store($(candidate_vsum_12_ptr) + r412_z, r412_value)
    r413_row = tl.arange($(96))
    r413_col = tl.arange($(512))
    r413_lane = tl.arange($(1024))
    for r413_group in range($(1)) {
      r413_left = tl.load($(candidate_vcat_71_ptr) + ((((((r413_row[:, None] * $(1024)) + r413_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + ((((r413_row[:, None] * $(1024)) + r413_lane[None, :]) % $(8)) * $(192)) + ((((r413_row[:, None] * $(1024)) + r413_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r413_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r413_lane[:, None] * $(512)) + r413_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r413_lane[:, None] * $(512)) + r413_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r413_lane[:, None] * $(512)) + r413_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r413_value = tl.dot(r413_left, r413_right)
      r413_value = r413_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_101_ptr) + ((r413_group * $(49152)) + (r413_row[:, None] * $(512)) + r413_col[None, :]), r413_value)
    }
    r414_row = tl.arange($(96))
    r414_col = tl.arange($(512))
    r414_lane = tl.arange($(1024))
    for r414_group in range($(1)) {
      r414_left = tl.load($(candidate_vcat_72_ptr) + ((((((r414_row[:, None] * $(1024)) + r414_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + ((((r414_row[:, None] * $(1024)) + r414_lane[None, :]) % $(8)) * $(128)) + ((((r414_row[:, None] * $(1024)) + r414_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r414_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r414_lane[:, None] * $(512)) + r414_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r414_lane[:, None] * $(512)) + r414_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r414_lane[:, None] * $(512)) + r414_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r414_value = tl.dot(r414_left, r414_right)
      r414_value = r414_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_102_ptr) + ((r414_group * $(49152)) + (r414_row[:, None] * $(512)) + r414_col[None, :]), r414_value)
    }
    r415_z = tl.arange($(49152))
    r415_value = (tl.load($(candidate_vbmm_101_ptr) + ((((r415_z / $(512)) % $(96)) * $(512)) + (r415_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_102_ptr) + ((((r415_z / $(512)) % $(96)) * $(512)) + (r415_z % $(512))), dtype=tl.bfloat16))
    r415_value = r415_value.to(tl.bfloat16)
    r415_value = tl.broadcast_to(r415_value, [$(49152)])
    tl.store($(candidate_vadd_57_ptr) + r415_z, r415_value)
    r416_z = tl.arange($(147456))
    r416_value = tl.where(((((r416_z / $(512)) % $(288)) >= $(96)) & (((r416_z / $(512)) % $(288)) < $(192)) & ((((r416_z / $(512)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vadd_57_ptr) + ((((((r416_z / $(512)) % $(288)) - $(96)) % $(96)) * $(512)) + (r416_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r416_value = r416_value.to(tl.bfloat16)
    r416_value = tl.broadcast_to(r416_value, [$(147456)])
    tl.store($(candidate_vslice_backward_20_ptr) + r416_z, r416_value)
    r417_z = tl.arange($(147456))
    r417_value = (tl.load($(candidate_vadd_51_ptr) + ((((r417_z / $(512)) % $(288)) * $(512)) + (r417_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_20_ptr) + ((((r417_z / $(512)) % $(288)) * $(512)) + (r417_z % $(512))), dtype=tl.bfloat16))
    r417_value = r417_value.to(tl.bfloat16)
    r417_value = tl.broadcast_to(r417_value, [$(147456)])
    tl.store($(candidate_vadd_58_ptr) + r417_z, r417_value)
    r418_z = tl.arange($(18432))
    r418_value = tl.where(((((r418_z / $(64)) % $(288)) >= $(96)) & (((r418_z / $(64)) % $(288)) < $(192)) & ((((r418_z / $(64)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vsum_12_ptr) + ((((((r418_z / $(64)) % $(288)) - $(96)) % $(96)) * $(64)) + (r418_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r418_value = r418_value.to(tl.bfloat16)
    r418_value = tl.broadcast_to(r418_value, [$(18432)])
    tl.store($(candidate_vslice_backward_21_ptr) + r418_z, r418_value)
    r419_z = tl.arange($(18432))
    r419_value = (tl.load($(candidate_vadd_52_ptr) + ((((r419_z / $(64)) % $(288)) * $(64)) + (r419_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_21_ptr) + ((((r419_z / $(64)) % $(288)) * $(64)) + (r419_z % $(64))), dtype=tl.bfloat16))
    r419_value = r419_value.to(tl.bfloat16)
    r419_value = tl.broadcast_to(r419_value, [$(18432)])
    tl.store($(candidate_vadd_59_ptr) + r419_z, r419_value)
    r420_z = tl.arange($(294912))
    r420_value = tl.where(((((r420_z / $(1024)) % $(288)) >= $(96)) & (((r420_z / $(1024)) % $(288)) < $(128)) & ((((r420_z / $(1024)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vcat_70_ptr) + ((((((r420_z / $(1024)) % $(288)) - $(96)) % $(32)) * $(1536)) + (((r420_z / $(128)) % $(8)) * $(192)) + (r420_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r420_value = r420_value.to(tl.bfloat16)
    r420_value = tl.broadcast_to(r420_value, [$(294912)])
    tl.store($(candidate_vslice_backward_22_ptr) + r420_z, r420_value)
    r421_z = tl.arange($(294912))
    r421_value = (tl.load($(candidate_vadd_53_ptr) + ((((r421_z / $(1024)) % $(288)) * $(1024)) + (((r421_z / $(128)) % $(8)) * $(128)) + (r421_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_22_ptr) + ((((r421_z / $(1024)) % $(288)) * $(1024)) + (((r421_z / $(128)) % $(8)) * $(128)) + (r421_z % $(128))), dtype=tl.bfloat16))
    r421_value = r421_value.to(tl.bfloat16)
    r421_value = tl.broadcast_to(r421_value, [$(294912)])
    tl.store($(candidate_vadd_60_ptr) + r421_z, r421_value)
    r422_z = tl.arange($(147456))
    r422_value = tl.where(((((r422_z / $(512)) % $(288)) >= $(96)) & (((r422_z / $(512)) % $(288)) < $(128)) & ((((r422_z / $(512)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vcat_70_ptr) + ((((((r422_z / $(512)) % $(288)) - $(96)) % $(32)) * $(1536)) + (((r422_z / $(64)) % $(8)) * $(192)) + (r422_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r422_value = r422_value.to(tl.bfloat16)
    r422_value = tl.broadcast_to(r422_value, [$(147456)])
    tl.store($(candidate_vslice_backward_23_ptr) + r422_z, r422_value)
    r423_z = tl.arange($(147456))
    r423_value = (tl.load($(candidate_vadd_54_ptr) + ((((r423_z / $(512)) % $(288)) * $(512)) + (((r423_z / $(64)) % $(8)) * $(64)) + (r423_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_23_ptr) + ((((r423_z / $(512)) % $(288)) * $(512)) + (((r423_z / $(64)) % $(8)) * $(64)) + (r423_z % $(64))), dtype=tl.bfloat16))
    r423_value = r423_value.to(tl.bfloat16)
    r423_value = tl.broadcast_to(r423_value, [$(147456)])
    tl.store($(candidate_vadd_61_ptr) + r423_z, r423_value)
    r424_row = tl.arange($(96))
    r424_col = tl.arange($(1024))
    r424_lane = tl.arange($(512))
    for r424_group in range($(1)) {
      r424_left = tl.load($(candidate_vcat_7_ptr) + ((((((r424_row[:, None] * $(512)) + r424_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r424_row[:, None] * $(512)) + r424_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r424_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r424_lane[:, None] * $(1024)) + r424_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r424_lane[:, None] * $(1024)) + r424_col[None, :]) % $(128)) * $(512)) + ((((r424_lane[:, None] * $(1024)) + r424_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r424_value = tl.dot(r424_left, r424_right)
      r424_value = r424_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_105_ptr) + ((r424_group * $(98304)) + (r424_row[:, None] * $(1024)) + r424_col[None, :]), r424_value)
    }
    r425_row = tl.arange($(96))
    r425_col = tl.arange($(1024))
    r425_lane = tl.arange($(512))
    for r425_group in range($(1)) {
      r425_left = tl.load($(candidate_vcat_7_ptr) + ((((((r425_row[:, None] * $(512)) + r425_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r425_row[:, None] * $(512)) + r425_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r425_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r425_lane[:, None] * $(1024)) + r425_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r425_lane[:, None] * $(1024)) + r425_col[None, :]) % $(128)) * $(512)) + ((((r425_lane[:, None] * $(1024)) + r425_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r425_value = tl.dot(r425_left, r425_right)
      r425_value = r425_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_106_ptr) + ((r425_group * $(98304)) + (r425_row[:, None] * $(1024)) + r425_col[None, :]), r425_value)
    }
    r426_z = tl.arange($(49152))
    r426_value = tl.where((r426_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r426_z / $(1536)) % $(32)) * $(1024)) + (((r426_z / $(192)) % $(8)) * $(128)) + ((r426_z % $(192)) - $(0)) + $(65536)) / $(1024)) % $(288)) * $(1536)) + (((((((r426_z / $(1536)) % $(32)) * $(1024)) + (((r426_z / $(192)) % $(8)) * $(128)) + ((r426_z % $(192)) - $(0)) + $(65536)) / $(128)) % $(8)) * $(192)) + (((((r426_z / $(1536)) % $(32)) * $(1024)) + (((r426_z / $(192)) % $(8)) * $(128)) + ((r426_z % $(192)) - $(0)) + $(65536)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r426_z / $(1536)) % $(32)) * $(512)) + (((r426_z / $(192)) % $(8)) * $(64)) + ((r426_z % $(192)) - $(128)) + $(32768)) / $(64)) % $(8)) * $(18432)) + (((((((r426_z / $(1536)) % $(32)) * $(512)) + (((r426_z / $(192)) % $(8)) * $(64)) + ((r426_z % $(192)) - $(128)) + $(32768)) / $(512)) % $(288)) * $(64)) + (((((r426_z / $(1536)) % $(32)) * $(512)) + (((r426_z / $(192)) % $(8)) * $(64)) + ((r426_z % $(192)) - $(128)) + $(32768)) % $(64))), dtype=tl.bfloat16))
    r426_value = r426_value.to(tl.bfloat16)
    r426_value = tl.broadcast_to(r426_value, [$(49152)])
    tl.store($(candidate_vcat_74_ptr) + r426_z, r426_value)
    r427_z = tl.arange($(147456))
    r427_value = tl.where((r427_z % $(192)) < $(128), tl.load($(candidate_vbmm_105_ptr) + ((((((((r427_z / $(1536)) % $(96)) * $(1024)) + (((r427_z / $(192)) % $(8)) * $(128)) + ((r427_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r427_z / $(1536)) % $(96)) * $(1024)) + (((r427_z / $(192)) % $(8)) * $(128)) + ((r427_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r427_z / $(1536)) % $(96)) * $(1024)) + (((r427_z / $(192)) % $(8)) * $(128)) + ((r427_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r427_z / $(1536)) % $(96)) * $(576)) + ((((r427_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r427_value = r427_value.to(tl.bfloat16)
    r427_value = tl.broadcast_to(r427_value, [$(147456)])
    tl.store($(candidate_vcat_75_ptr) + r427_z, r427_value)
    r428_z = tl.arange($(49152))
    r428_value = tl.load($(candidate_vcat_74_ptr) + ((((r428_z / $(192)) % $(32)) * $(1536)) + (((r428_z / $(6144)) % $(8)) * $(192)) + (r428_z % $(192))), dtype=tl.bfloat16)
    r428_value = r428_value.to(tl.float32)
    r428_value = tl.broadcast_to(r428_value, [$(49152)])
    tl.store($(candidate_vto_copy_133_ptr) + r428_z, r428_value)
    r429_z = tl.arange($(147456))
    r429_value = tl.load($(candidate_vcat_75_ptr) + ((((r429_z / $(192)) % $(96)) * $(1536)) + (((r429_z / $(18432)) % $(8)) * $(192)) + (r429_z % $(192))), dtype=tl.bfloat16)
    r429_value = r429_value.to(tl.float32)
    r429_value = tl.broadcast_to(r429_value, [$(147456)])
    tl.store($(candidate_vto_copy_134_ptr) + r429_z, r429_value)
    r430_row = tl.arange($(32))
    r430_col = tl.arange($(96))
    r430_lane = tl.arange($(192))
    for r430_group in range($(8)) {
      r430_left = tl.load($(candidate_vto_copy_133_ptr) + (((((((r430_group % $(8)) * $(6144)) + (r430_row[:, None] * $(192)) + r430_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r430_group % $(8)) * $(6144)) + (r430_row[:, None] * $(192)) + r430_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r430_group % $(8)) * $(6144)) + (r430_row[:, None] * $(192)) + r430_lane[None, :]) % $(192))), dtype=tl.float32)
      r430_right = tl.load($(candidate_vto_copy_134_ptr) + (((((((r430_group % $(8)) * $(18432)) + (r430_lane[:, None] * $(96)) + r430_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r430_group % $(8)) * $(18432)) + (r430_lane[:, None] * $(96)) + r430_col[None, :]) % $(96)) * $(192)) + (((((r430_group % $(8)) * $(18432)) + (r430_lane[:, None] * $(96)) + r430_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r430_value = tl.dot(r430_left, r430_right)
      r430_value = r430_value.to(tl.float32)
      tl.store($(candidate_vbmm_107_ptr) + ((r430_group * $(3072)) + (r430_row[:, None] * $(96)) + r430_col[None, :]), r430_value)
    }
    r431_z = tl.arange($(24576))
    r431_value = (tl.load($(candidate_vbmm_107_ptr) + ((((r431_z / $(3072)) % $(8)) * $(3072)) + (((r431_z / $(96)) % $(32)) * $(96)) + (r431_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r431_value = r431_value.to(tl.float32)
    r431_value = tl.broadcast_to(r431_value, [$(24576)])
    tl.store($(candidate_vmul_66_ptr) + r431_z, r431_value)
    r432_z = tl.arange($(32))
    r432_value = (tl.cast(r432_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r432_value = tl.broadcast_to(r432_value, [$(32)])
    tl.store($(candidate_vadd_62_ptr) + r432_z, r432_value)
    r433_z = tl.arange($(32))
    r433_value = (tl.load($(candidate_vadd_62_ptr) + r433_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r433_value = tl.broadcast_to(r433_value, [$(32)])
    tl.store($(candidate_vsub_48_ptr) + r433_z, r433_value)
    r434_z = tl.arange($(3072))
    r434_value = (tl.cast((r434_z % $(96)), tl.int64) > tl.load($(candidate_vsub_48_ptr) + ((r434_z / $(96)) % $(32)), dtype=tl.int64))
    r434_value = tl.broadcast_to(r434_value, [$(3072)])
    tl.store($(candidate_vgt_21_ptr) + r434_z, r434_value)
    r435_z = tl.arange($(24576))
    r435_value = tl.where(tl.load($(candidate_vgt_21_ptr) + ((((r435_z / $(96)) % $(32)) * $(96)) + (r435_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_66_ptr) + ((((r435_z / $(3072)) % $(8)) * $(3072)) + (((r435_z / $(96)) % $(32)) * $(96)) + (r435_z % $(96))), dtype=tl.float32))
    r435_value = r435_value.to(tl.float32)
    r435_value = tl.broadcast_to(r435_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_21_ptr) + r435_z, r435_value)
    r436_z = tl.arange($(24576))
    r436_lane = tl.arange($(96))
    r436_row_values = tl.load($(candidate_vmasked_fill_21_ptr) + ((((r436_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r436_z[:, None] / $(96)) % $(32)) * $(96)) + (r436_lane[None, :] % $(96))), dtype=tl.float32)
    r436_maximum = tl.max(r436_row_values, axis=1, keep_dims=true)
    r436_weights = tl.exp(r436_row_values - r436_maximum)
    r436_total = tl.sum(r436_weights, axis=1)
    r436_maximum = tl.max(r436_row_values, axis=1)
    r436_value = tl.exp(tl.load($(candidate_vmasked_fill_21_ptr) + ((((r436_z / $(3072)) % $(8)) * $(3072)) + (((r436_z / $(96)) % $(32)) * $(96)) + (r436_z % $(96))), dtype=tl.float32) - r436_maximum) / r436_total
    r436_value = r436_value.to(tl.float32)
    r436_value = tl.broadcast_to(r436_value, [$(24576)])
    tl.store($(candidate_vsoftmax_15_ptr) + r436_z, r436_value)
    r437_z = tl.arange($(24576))
    r437_value = tl.load($(candidate_vsoftmax_15_ptr) + ((((r437_z / $(3072)) % $(8)) * $(3072)) + (((r437_z / $(96)) % $(32)) * $(96)) + (r437_z % $(96))), dtype=tl.float32)
    r437_value = r437_value.to(tl.bfloat16)
    r437_value = tl.broadcast_to(r437_value, [$(24576)])
    tl.store($(candidate_vto_copy_135_ptr) + r437_z, r437_value)
    r438_row = tl.arange($(32))
    r438_col = tl.arange($(128))
    r438_lane = tl.arange($(96))
    for r438_group in range($(8)) {
      r438_left = tl.load($(candidate_vto_copy_135_ptr) + (((((((r438_group % $(8)) * $(3072)) + (r438_row[:, None] * $(96)) + r438_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r438_group % $(8)) * $(3072)) + (r438_row[:, None] * $(96)) + r438_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r438_group % $(8)) * $(3072)) + (r438_row[:, None] * $(96)) + r438_lane[None, :]) % $(96))), dtype=tl.bfloat16)
      r438_right = tl.load($(candidate_vbmm_106_ptr) + (((((((r438_group % $(8)) * $(12288)) + (r438_lane[:, None] * $(128)) + r438_col[None, :]) / $(128)) % $(96)) * $(1024)) + ((((((r438_group % $(8)) * $(12288)) + (r438_lane[:, None] * $(128)) + r438_col[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r438_group % $(8)) * $(12288)) + (r438_lane[:, None] * $(128)) + r438_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r438_value = tl.dot(r438_left, r438_right)
      r438_value = r438_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_108_ptr) + ((r438_group * $(4096)) + (r438_row[:, None] * $(128)) + r438_col[None, :]), r438_value)
    }
    r439_z = tl.arange($(256))
    r439_lane = tl.arange($(96))
    r439_row_values = tl.load($(candidate_vmasked_fill_21_ptr) + ((((r439_z[:, None] / $(32)) % $(8)) * $(3072)) + ((r439_z[:, None] % $(32)) * $(96)) + (r439_lane[None, :] % $(96))), dtype=tl.float32)
    r439_maximum = tl.max(r439_row_values, axis=1, keep_dims=true)
    r439_weights = tl.exp(r439_row_values - r439_maximum)
    r439_total = tl.sum(r439_weights, axis=1)
    r439_maximum = tl.max(r439_row_values, axis=1)
    r439_value = r439_maximum + tl.log(r439_total)
    r439_value = r439_value.to(tl.float32)
    r439_value = tl.broadcast_to(r439_value, [$(256)])
    tl.store($(candidate_vlogsumexp_15_ptr) + r439_z, r439_value)
    r440_z = tl.arange($(32768))
    r440_value = tl.load($(candidate_vbmm_108_ptr) + ((((((((r440_z / $(128)) % $(8)) * $(4096)) + ((((r440_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r440_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r440_z / $(128)) % $(8)) * $(32)) + (((r440_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r440_z % $(128))), dtype=tl.bfloat16)
    r440_value = r440_value.to(tl.bfloat16)
    r440_value = tl.broadcast_to(r440_value, [$(32768)])
    tl.store($(candidate_vcat_76_ptr) + r440_z, r440_value)
    r441_z = tl.arange($(256))
    r441_value = tl.load($(candidate_vlogsumexp_15_ptr) + (((r441_z % $(8)) * $(32)) + ((((r441_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r441_value = r441_value.to(tl.float32)
    r441_value = tl.broadcast_to(r441_value, [$(256)])
    tl.store($(candidate_vcat_77_ptr) + r441_z, r441_value)
    r442_z = tl.arange($(49152))
    r442_value = tl.load($(candidate_vcat_74_ptr) + ((((r442_z / $(192)) % $(32)) * $(1536)) + (((r442_z / $(6144)) % $(8)) * $(192)) + (r442_z % $(192))), dtype=tl.bfloat16)
    r442_value = r442_value.to(tl.float32)
    r442_value = tl.broadcast_to(r442_value, [$(49152)])
    tl.store($(candidate_vto_copy_136_ptr) + r442_z, r442_value)
    r443_z = tl.arange($(147456))
    r443_value = tl.load($(candidate_vcat_75_ptr) + ((((r443_z / $(192)) % $(96)) * $(1536)) + (((r443_z / $(18432)) % $(8)) * $(192)) + (r443_z % $(192))), dtype=tl.bfloat16)
    r443_value = r443_value.to(tl.float32)
    r443_value = tl.broadcast_to(r443_value, [$(147456)])
    tl.store($(candidate_vto_copy_137_ptr) + r443_z, r443_value)
    r444_z = tl.arange($(98304))
    r444_value = tl.load($(candidate_vbmm_106_ptr) + ((((r444_z / $(128)) % $(96)) * $(1024)) + (((r444_z / $(12288)) % $(8)) * $(128)) + (r444_z % $(128))), dtype=tl.bfloat16)
    r444_value = r444_value.to(tl.float32)
    r444_value = tl.broadcast_to(r444_value, [$(98304)])
    tl.store($(candidate_vto_copy_138_ptr) + r444_z, r444_value)
    r445_z = tl.arange($(32768))
    r445_value = tl.load($(candidate_vmm_22_ptr) + ((((r445_z / $(128)) % $(32)) * $(1024)) + (((r445_z / $(4096)) % $(8)) * $(128)) + (r445_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r445_value = r445_value.to(tl.float32)
    r445_value = tl.broadcast_to(r445_value, [$(32768)])
    tl.store($(candidate_vto_copy_139_ptr) + r445_z, r445_value)
    r446_z = tl.arange($(32768))
    r446_value = tl.load($(candidate_vcat_76_ptr) + ((((r446_z / $(128)) % $(32)) * $(1024)) + (((r446_z / $(4096)) % $(8)) * $(128)) + (r446_z % $(128))), dtype=tl.bfloat16)
    r446_value = r446_value.to(tl.float32)
    r446_value = tl.broadcast_to(r446_value, [$(32768)])
    tl.store($(candidate_vto_copy_140_ptr) + r446_z, r446_value)
    r447_row = tl.arange($(32))
    r447_col = tl.arange($(96))
    r447_lane = tl.arange($(192))
    for r447_group in range($(8)) {
      r447_left = tl.load($(candidate_vto_copy_136_ptr) + (((((((r447_group % $(8)) * $(6144)) + (r447_row[:, None] * $(192)) + r447_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r447_group % $(8)) * $(6144)) + (r447_row[:, None] * $(192)) + r447_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r447_group % $(8)) * $(6144)) + (r447_row[:, None] * $(192)) + r447_lane[None, :]) % $(192))), dtype=tl.float32)
      r447_right = tl.load($(candidate_vto_copy_137_ptr) + (((((((r447_group % $(8)) * $(18432)) + (r447_lane[:, None] * $(96)) + r447_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r447_group % $(8)) * $(18432)) + (r447_lane[:, None] * $(96)) + r447_col[None, :]) % $(96)) * $(192)) + (((((r447_group % $(8)) * $(18432)) + (r447_lane[:, None] * $(96)) + r447_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r447_value = tl.dot(r447_left, r447_right)
      r447_value = r447_value.to(tl.float32)
      tl.store($(candidate_vbmm_109_ptr) + ((r447_group * $(3072)) + (r447_row[:, None] * $(96)) + r447_col[None, :]), r447_value)
    }
    r448_z = tl.arange($(24576))
    r448_value = (tl.load($(candidate_vbmm_109_ptr) + ((((r448_z / $(3072)) % $(8)) * $(3072)) + (((r448_z / $(96)) % $(32)) * $(96)) + (r448_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r448_value = r448_value.to(tl.float32)
    r448_value = tl.broadcast_to(r448_value, [$(24576)])
    tl.store($(candidate_vmul_67_ptr) + r448_z, r448_value)
    r449_z = tl.arange($(32))
    r449_value = (tl.cast(r449_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r449_value = tl.broadcast_to(r449_value, [$(32)])
    tl.store($(candidate_vadd_63_ptr) + r449_z, r449_value)
    r450_z = tl.arange($(32))
    r450_value = (tl.load($(candidate_vadd_63_ptr) + r450_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r450_value = tl.broadcast_to(r450_value, [$(32)])
    tl.store($(candidate_vsub_49_ptr) + r450_z, r450_value)
    r451_z = tl.arange($(3072))
    r451_value = (tl.cast((r451_z % $(96)), tl.int64) > tl.load($(candidate_vsub_49_ptr) + ((r451_z / $(96)) % $(32)), dtype=tl.int64))
    r451_value = tl.broadcast_to(r451_value, [$(3072)])
    tl.store($(candidate_vgt_22_ptr) + r451_z, r451_value)
    r452_z = tl.arange($(24576))
    r452_value = tl.where(tl.load($(candidate_vgt_22_ptr) + ((((r452_z / $(96)) % $(32)) * $(96)) + (r452_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_67_ptr) + ((((r452_z / $(3072)) % $(8)) * $(3072)) + (((r452_z / $(96)) % $(32)) * $(96)) + (r452_z % $(96))), dtype=tl.float32))
    r452_value = r452_value.to(tl.float32)
    r452_value = tl.broadcast_to(r452_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_22_ptr) + r452_z, r452_value)
    r453_z = tl.arange($(24576))
    r453_value = (tl.load($(candidate_vmasked_fill_22_ptr) + ((((r453_z / $(3072)) % $(8)) * $(3072)) + (((r453_z / $(96)) % $(32)) * $(96)) + (r453_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vcat_77_ptr) + ((((r453_z / $(96)) % $(32)) * $(8)) + ((r453_z / $(3072)) % $(8))), dtype=tl.float32))
    r453_value = r453_value.to(tl.float32)
    r453_value = tl.broadcast_to(r453_value, [$(24576)])
    tl.store($(candidate_vsub_50_ptr) + r453_z, r453_value)
    r454_z = tl.arange($(24576))
    r454_value = tl.exp(tl.load($(candidate_vsub_50_ptr) + ((((r454_z / $(3072)) % $(8)) * $(3072)) + (((r454_z / $(96)) % $(32)) * $(96)) + (r454_z % $(96))), dtype=tl.float32))
    r454_value = r454_value.to(tl.float32)
    r454_value = tl.broadcast_to(r454_value, [$(24576)])
    tl.store($(candidate_vexp_6_ptr) + r454_z, r454_value)
    r455_z = tl.arange($(32768))
    r455_value = (tl.load($(candidate_vto_copy_139_ptr) + ((((r455_z / $(4096)) % $(8)) * $(4096)) + (((r455_z / $(128)) % $(32)) * $(128)) + (r455_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_140_ptr) + ((((r455_z / $(4096)) % $(8)) * $(4096)) + (((r455_z / $(128)) % $(32)) * $(128)) + (r455_z % $(128))), dtype=tl.float32))
    r455_value = r455_value.to(tl.float32)
    r455_value = tl.broadcast_to(r455_value, [$(32768)])
    tl.store($(candidate_vmul_68_ptr) + r455_z, r455_value)
    r456_z = tl.arange($(256))
    r456_lane = tl.arange($(128))
    r456_value = tl.sum(tl.load($(candidate_vmul_68_ptr) + ((((r456_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r456_z[:, None] % $(32)) * $(128)) + r456_lane[None, :]), dtype=tl.float32), axis=1)
    r456_value = r456_value.to(tl.float32)
    r456_value = tl.broadcast_to(r456_value, [$(256)])
    tl.store($(candidate_vsum_13_ptr) + r456_z, r456_value)
    r457_row = tl.arange($(32))
    r457_col = tl.arange($(96))
    r457_lane = tl.arange($(128))
    for r457_group in range($(8)) {
      r457_left = tl.load($(candidate_vto_copy_139_ptr) + (((((((r457_group % $(8)) * $(4096)) + (r457_row[:, None] * $(128)) + r457_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r457_group % $(8)) * $(4096)) + (r457_row[:, None] * $(128)) + r457_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r457_group % $(8)) * $(4096)) + (r457_row[:, None] * $(128)) + r457_lane[None, :]) % $(128))), dtype=tl.float32)
      r457_right = tl.load($(candidate_vto_copy_138_ptr) + (((((((r457_group % $(8)) * $(12288)) + (r457_lane[:, None] * $(96)) + r457_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r457_group % $(8)) * $(12288)) + (r457_lane[:, None] * $(96)) + r457_col[None, :]) % $(96)) * $(128)) + (((((r457_group % $(8)) * $(12288)) + (r457_lane[:, None] * $(96)) + r457_col[None, :]) / $(96)) % $(128))), dtype=tl.float32)
      r457_value = tl.dot(r457_left, r457_right)
      r457_value = r457_value.to(tl.float32)
      tl.store($(candidate_vbmm_110_ptr) + ((r457_group * $(3072)) + (r457_row[:, None] * $(96)) + r457_col[None, :]), r457_value)
    }
    r458_z = tl.arange($(24576))
    r458_value = (tl.load($(candidate_vbmm_110_ptr) + ((((r458_z / $(3072)) % $(8)) * $(3072)) + (((r458_z / $(96)) % $(32)) * $(96)) + (r458_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vsum_13_ptr) + ((((r458_z / $(3072)) % $(8)) * $(32)) + ((r458_z / $(96)) % $(32))), dtype=tl.float32))
    r458_value = r458_value.to(tl.float32)
    r458_value = tl.broadcast_to(r458_value, [$(24576)])
    tl.store($(candidate_vsub_51_ptr) + r458_z, r458_value)
    r459_z = tl.arange($(24576))
    r459_value = (tl.load($(candidate_vexp_6_ptr) + ((((r459_z / $(3072)) % $(8)) * $(3072)) + (((r459_z / $(96)) % $(32)) * $(96)) + (r459_z % $(96))), dtype=tl.float32) * tl.load($(candidate_vsub_51_ptr) + ((((r459_z / $(3072)) % $(8)) * $(3072)) + (((r459_z / $(96)) % $(32)) * $(96)) + (r459_z % $(96))), dtype=tl.float32))
    r459_value = r459_value.to(tl.float32)
    r459_value = tl.broadcast_to(r459_value, [$(24576)])
    tl.store($(candidate_vmul_69_ptr) + r459_z, r459_value)
    r460_row = tl.arange($(32))
    r460_col = tl.arange($(192))
    r460_lane = tl.arange($(96))
    for r460_group in range($(8)) {
      r460_left = tl.load($(candidate_vmul_69_ptr) + (((((((r460_group % $(8)) * $(3072)) + (r460_row[:, None] * $(96)) + r460_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r460_group % $(8)) * $(3072)) + (r460_row[:, None] * $(96)) + r460_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r460_group % $(8)) * $(3072)) + (r460_row[:, None] * $(96)) + r460_lane[None, :]) % $(96))), dtype=tl.float32)
      r460_right = tl.load($(candidate_vto_copy_137_ptr) + (((((((r460_group % $(8)) * $(18432)) + (r460_lane[:, None] * $(192)) + r460_col[None, :]) / $(18432)) % $(8)) * $(18432)) + ((((((r460_group % $(8)) * $(18432)) + (r460_lane[:, None] * $(192)) + r460_col[None, :]) / $(192)) % $(96)) * $(192)) + ((((r460_group % $(8)) * $(18432)) + (r460_lane[:, None] * $(192)) + r460_col[None, :]) % $(192))), dtype=tl.float32)
      r460_value = tl.dot(r460_left, r460_right)
      r460_value = r460_value.to(tl.float32)
      tl.store($(candidate_vbmm_111_ptr) + ((r460_group * $(6144)) + (r460_row[:, None] * $(192)) + r460_col[None, :]), r460_value)
    }
    r461_z = tl.arange($(49152))
    r461_value = (tl.load($(candidate_vbmm_111_ptr) + ((((r461_z / $(6144)) % $(8)) * $(6144)) + (((r461_z / $(192)) % $(32)) * $(192)) + (r461_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r461_value = r461_value.to(tl.float32)
    r461_value = tl.broadcast_to(r461_value, [$(49152)])
    tl.store($(candidate_vmul_70_ptr) + r461_z, r461_value)
    r462_z = tl.arange($(49152))
    r462_value = tl.load($(candidate_vmul_70_ptr) + ((((r462_z / $(192)) % $(8)) * $(6144)) + (((r462_z / $(1536)) % $(32)) * $(192)) + (r462_z % $(192))), dtype=tl.float32)
    r462_value = r462_value.to(tl.bfloat16)
    r462_value = tl.broadcast_to(r462_value, [$(49152)])
    tl.store($(candidate_vto_copy_141_ptr) + r462_z, r462_value)
    r463_row = tl.arange($(96))
    r463_col = tl.arange($(192))
    r463_lane = tl.arange($(32))
    for r463_group in range($(8)) {
      r463_left = tl.load($(candidate_vmul_69_ptr) + (((((((r463_group % $(8)) * $(3072)) + (r463_row[:, None] * $(32)) + r463_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r463_group % $(8)) * $(3072)) + (r463_row[:, None] * $(32)) + r463_lane[None, :]) % $(32)) * $(96)) + (((((r463_group % $(8)) * $(3072)) + (r463_row[:, None] * $(32)) + r463_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r463_right = tl.load($(candidate_vto_copy_136_ptr) + (((((((r463_group % $(8)) * $(6144)) + (r463_lane[:, None] * $(192)) + r463_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r463_group % $(8)) * $(6144)) + (r463_lane[:, None] * $(192)) + r463_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r463_group % $(8)) * $(6144)) + (r463_lane[:, None] * $(192)) + r463_col[None, :]) % $(192))), dtype=tl.float32)
      r463_value = tl.dot(r463_left, r463_right)
      r463_value = r463_value.to(tl.float32)
      tl.store($(candidate_vbmm_112_ptr) + ((r463_group * $(18432)) + (r463_row[:, None] * $(192)) + r463_col[None, :]), r463_value)
    }
    r464_z = tl.arange($(147456))
    r464_value = (tl.load($(candidate_vbmm_112_ptr) + ((((r464_z / $(18432)) % $(8)) * $(18432)) + (((r464_z / $(192)) % $(96)) * $(192)) + (r464_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r464_value = r464_value.to(tl.float32)
    r464_value = tl.broadcast_to(r464_value, [$(147456)])
    tl.store($(candidate_vmul_71_ptr) + r464_z, r464_value)
    r465_z = tl.arange($(147456))
    r465_value = tl.load($(candidate_vmul_71_ptr) + ((((r465_z / $(192)) % $(8)) * $(18432)) + (((r465_z / $(1536)) % $(96)) * $(192)) + (r465_z % $(192))), dtype=tl.float32)
    r465_value = r465_value.to(tl.bfloat16)
    r465_value = tl.broadcast_to(r465_value, [$(147456)])
    tl.store($(candidate_vto_copy_142_ptr) + r465_z, r465_value)
    r466_row = tl.arange($(96))
    r466_col = tl.arange($(128))
    r466_lane = tl.arange($(32))
    for r466_group in range($(8)) {
      r466_left = tl.load($(candidate_vexp_6_ptr) + (((((((r466_group % $(8)) * $(3072)) + (r466_row[:, None] * $(32)) + r466_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r466_group % $(8)) * $(3072)) + (r466_row[:, None] * $(32)) + r466_lane[None, :]) % $(32)) * $(96)) + (((((r466_group % $(8)) * $(3072)) + (r466_row[:, None] * $(32)) + r466_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r466_right = tl.load($(candidate_vto_copy_139_ptr) + (((((((r466_group % $(8)) * $(4096)) + (r466_lane[:, None] * $(128)) + r466_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r466_group % $(8)) * $(4096)) + (r466_lane[:, None] * $(128)) + r466_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r466_group % $(8)) * $(4096)) + (r466_lane[:, None] * $(128)) + r466_col[None, :]) % $(128))), dtype=tl.float32)
      r466_value = tl.dot(r466_left, r466_right)
      r466_value = r466_value.to(tl.float32)
      tl.store($(candidate_vbmm_113_ptr) + ((r466_group * $(12288)) + (r466_row[:, None] * $(128)) + r466_col[None, :]), r466_value)
    }
    r467_z = tl.arange($(98304))
    r467_value = tl.load($(candidate_vbmm_113_ptr) + ((((r467_z / $(128)) % $(8)) * $(12288)) + (((r467_z / $(1024)) % $(96)) * $(128)) + (r467_z % $(128))), dtype=tl.float32)
    r467_value = r467_value.to(tl.bfloat16)
    r467_value = tl.broadcast_to(r467_value, [$(98304)])
    tl.store($(candidate_vto_copy_143_ptr) + r467_z, r467_value)
    r468_z = tl.arange($(49152))
    r468_value = tl.load($(candidate_vto_copy_141_ptr) + ((((((r468_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(1536)) + (((r468_z / $(192)) % $(8)) * $(192)) + (r468_z % $(192))), dtype=tl.bfloat16)
    r468_value = r468_value.to(tl.bfloat16)
    r468_value = tl.broadcast_to(r468_value, [$(49152)])
    tl.store($(candidate_vcat_78_ptr) + r468_z, r468_value)
    r469_z = tl.arange($(147456))
    r469_value = tl.load($(candidate_vto_copy_142_ptr) + ((((((r469_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r469_z / $(192)) % $(8)) * $(192)) + (r469_z % $(192))), dtype=tl.bfloat16)
    r469_value = r469_value.to(tl.bfloat16)
    r469_value = tl.broadcast_to(r469_value, [$(147456)])
    tl.store($(candidate_vcat_79_ptr) + r469_z, r469_value)
    r470_z = tl.arange($(98304))
    r470_value = tl.load($(candidate_vto_copy_143_ptr) + ((((((r470_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r470_z / $(128)) % $(8)) * $(128)) + (r470_z % $(128))), dtype=tl.bfloat16)
    r470_value = r470_value.to(tl.bfloat16)
    r470_value = tl.broadcast_to(r470_value, [$(98304)])
    tl.store($(candidate_vcat_80_ptr) + r470_z, r470_value)
    r471_z = tl.arange($(6144))
    r471_lane = tl.arange($(8))
    r471_value = tl.sum(tl.load($(candidate_vcat_79_ptr) + ((((r471_z[:, None] / $(64)) % $(96)) * $(1536)) + (r471_lane[None, :] * $(192)) + (r471_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r471_value = r471_value.to(tl.bfloat16)
    r471_value = tl.broadcast_to(r471_value, [$(6144)])
    tl.store($(candidate_vsum_14_ptr) + r471_z, r471_value)
    r472_row = tl.arange($(96))
    r472_col = tl.arange($(512))
    r472_lane = tl.arange($(1024))
    for r472_group in range($(1)) {
      r472_left = tl.load($(candidate_vcat_79_ptr) + ((((((r472_row[:, None] * $(1024)) + r472_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + ((((r472_row[:, None] * $(1024)) + r472_lane[None, :]) % $(8)) * $(192)) + ((((r472_row[:, None] * $(1024)) + r472_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r472_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r472_lane[:, None] * $(512)) + r472_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r472_lane[:, None] * $(512)) + r472_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r472_lane[:, None] * $(512)) + r472_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r472_value = tl.dot(r472_left, r472_right)
      r472_value = r472_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_114_ptr) + ((r472_group * $(49152)) + (r472_row[:, None] * $(512)) + r472_col[None, :]), r472_value)
    }
    r473_row = tl.arange($(96))
    r473_col = tl.arange($(512))
    r473_lane = tl.arange($(1024))
    for r473_group in range($(1)) {
      r473_left = tl.load($(candidate_vcat_80_ptr) + ((((((r473_row[:, None] * $(1024)) + r473_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + ((((r473_row[:, None] * $(1024)) + r473_lane[None, :]) % $(8)) * $(128)) + ((((r473_row[:, None] * $(1024)) + r473_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r473_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r473_lane[:, None] * $(512)) + r473_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r473_lane[:, None] * $(512)) + r473_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r473_lane[:, None] * $(512)) + r473_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r473_value = tl.dot(r473_left, r473_right)
      r473_value = r473_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_115_ptr) + ((r473_group * $(49152)) + (r473_row[:, None] * $(512)) + r473_col[None, :]), r473_value)
    }
    r474_z = tl.arange($(49152))
    r474_value = (tl.load($(candidate_vbmm_114_ptr) + ((((r474_z / $(512)) % $(96)) * $(512)) + (r474_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_115_ptr) + ((((r474_z / $(512)) % $(96)) * $(512)) + (r474_z % $(512))), dtype=tl.bfloat16))
    r474_value = r474_value.to(tl.bfloat16)
    r474_value = tl.broadcast_to(r474_value, [$(49152)])
    tl.store($(candidate_vadd_64_ptr) + r474_z, r474_value)
    r475_z = tl.arange($(147456))
    r475_value = tl.where(((((r475_z / $(512)) % $(288)) >= $(0)) & (((r475_z / $(512)) % $(288)) < $(96)) & ((((r475_z / $(512)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_64_ptr) + ((((((r475_z / $(512)) % $(288)) - $(0)) % $(96)) * $(512)) + (r475_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r475_value = r475_value.to(tl.bfloat16)
    r475_value = tl.broadcast_to(r475_value, [$(147456)])
    tl.store($(candidate_vslice_backward_24_ptr) + r475_z, r475_value)
    r476_z = tl.arange($(147456))
    r476_value = (tl.load($(candidate_vadd_58_ptr) + ((((r476_z / $(512)) % $(288)) * $(512)) + (r476_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_24_ptr) + ((((r476_z / $(512)) % $(288)) * $(512)) + (r476_z % $(512))), dtype=tl.bfloat16))
    r476_value = r476_value.to(tl.bfloat16)
    r476_value = tl.broadcast_to(r476_value, [$(147456)])
    tl.store($(candidate_vadd_65_ptr) + r476_z, r476_value)
    r477_z = tl.arange($(18432))
    r477_value = tl.where(((((r477_z / $(64)) % $(288)) >= $(0)) & (((r477_z / $(64)) % $(288)) < $(96)) & ((((r477_z / $(64)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_14_ptr) + ((((((r477_z / $(64)) % $(288)) - $(0)) % $(96)) * $(64)) + (r477_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r477_value = r477_value.to(tl.bfloat16)
    r477_value = tl.broadcast_to(r477_value, [$(18432)])
    tl.store($(candidate_vslice_backward_25_ptr) + r477_z, r477_value)
    r478_z = tl.arange($(18432))
    r478_value = (tl.load($(candidate_vadd_59_ptr) + ((((r478_z / $(64)) % $(288)) * $(64)) + (r478_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_25_ptr) + ((((r478_z / $(64)) % $(288)) * $(64)) + (r478_z % $(64))), dtype=tl.bfloat16))
    r478_value = r478_value.to(tl.bfloat16)
    r478_value = tl.broadcast_to(r478_value, [$(18432)])
    tl.store($(candidate_vadd_66_ptr) + r478_z, r478_value)
    r479_z = tl.arange($(294912))
    r479_value = tl.where(((((r479_z / $(1024)) % $(288)) >= $(64)) & (((r479_z / $(1024)) % $(288)) < $(96)) & ((((r479_z / $(1024)) % $(288)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vcat_78_ptr) + ((((((r479_z / $(1024)) % $(288)) - $(64)) % $(32)) * $(1536)) + (((r479_z / $(128)) % $(8)) * $(192)) + (r479_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r479_value = r479_value.to(tl.bfloat16)
    r479_value = tl.broadcast_to(r479_value, [$(294912)])
    tl.store($(candidate_vslice_backward_26_ptr) + r479_z, r479_value)
    r480_z = tl.arange($(294912))
    r480_value = (tl.load($(candidate_vadd_60_ptr) + ((((r480_z / $(1024)) % $(288)) * $(1024)) + (((r480_z / $(128)) % $(8)) * $(128)) + (r480_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_26_ptr) + ((((r480_z / $(1024)) % $(288)) * $(1024)) + (((r480_z / $(128)) % $(8)) * $(128)) + (r480_z % $(128))), dtype=tl.bfloat16))
    r480_value = r480_value.to(tl.bfloat16)
    r480_value = tl.broadcast_to(r480_value, [$(294912)])
    tl.store($(candidate_vadd_67_ptr) + r480_z, r480_value)
    r481_z = tl.arange($(147456))
    r481_value = tl.where(((((r481_z / $(512)) % $(288)) >= $(64)) & (((r481_z / $(512)) % $(288)) < $(96)) & ((((r481_z / $(512)) % $(288)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vcat_78_ptr) + ((((((r481_z / $(512)) % $(288)) - $(64)) % $(32)) * $(1536)) + (((r481_z / $(64)) % $(8)) * $(192)) + (r481_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r481_value = r481_value.to(tl.bfloat16)
    r481_value = tl.broadcast_to(r481_value, [$(147456)])
    tl.store($(candidate_vslice_backward_27_ptr) + r481_z, r481_value)
    r482_z = tl.arange($(147456))
    r482_value = (tl.load($(candidate_vadd_61_ptr) + ((((r482_z / $(512)) % $(288)) * $(512)) + (((r482_z / $(64)) % $(8)) * $(64)) + (r482_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_27_ptr) + ((((r482_z / $(512)) % $(288)) * $(512)) + (((r482_z / $(64)) % $(8)) * $(64)) + (r482_z % $(64))), dtype=tl.bfloat16))
    r482_value = r482_value.to(tl.bfloat16)
    r482_value = tl.broadcast_to(r482_value, [$(147456)])
    tl.store($(candidate_vadd_68_ptr) + r482_z, r482_value)
    r483_row = tl.arange($(96))
    r483_col = tl.arange($(1024))
    r483_lane = tl.arange($(512))
    for r483_group in range($(1)) {
      r483_left = tl.load($(candidate_vcat_7_ptr) + ((((((r483_row[:, None] * $(512)) + r483_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r483_row[:, None] * $(512)) + r483_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r483_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r483_lane[:, None] * $(1024)) + r483_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r483_lane[:, None] * $(1024)) + r483_col[None, :]) % $(128)) * $(512)) + ((((r483_lane[:, None] * $(1024)) + r483_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r483_value = tl.dot(r483_left, r483_right)
      r483_value = r483_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_118_ptr) + ((r483_group * $(98304)) + (r483_row[:, None] * $(1024)) + r483_col[None, :]), r483_value)
    }
    r484_row = tl.arange($(96))
    r484_col = tl.arange($(1024))
    r484_lane = tl.arange($(512))
    for r484_group in range($(1)) {
      r484_left = tl.load($(candidate_vcat_7_ptr) + ((((((r484_row[:, None] * $(512)) + r484_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r484_row[:, None] * $(512)) + r484_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r484_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r484_lane[:, None] * $(1024)) + r484_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r484_lane[:, None] * $(1024)) + r484_col[None, :]) % $(128)) * $(512)) + ((((r484_lane[:, None] * $(1024)) + r484_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r484_value = tl.dot(r484_left, r484_right)
      r484_value = r484_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_119_ptr) + ((r484_group * $(98304)) + (r484_row[:, None] * $(1024)) + r484_col[None, :]), r484_value)
    }
    r485_z = tl.arange($(49152))
    r485_value = tl.where((r485_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r485_z / $(1536)) % $(32)) * $(1024)) + (((r485_z / $(192)) % $(8)) * $(128)) + ((r485_z % $(192)) - $(0)) + $(32768)) / $(1024)) % $(288)) * $(1536)) + (((((((r485_z / $(1536)) % $(32)) * $(1024)) + (((r485_z / $(192)) % $(8)) * $(128)) + ((r485_z % $(192)) - $(0)) + $(32768)) / $(128)) % $(8)) * $(192)) + (((((r485_z / $(1536)) % $(32)) * $(1024)) + (((r485_z / $(192)) % $(8)) * $(128)) + ((r485_z % $(192)) - $(0)) + $(32768)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r485_z / $(1536)) % $(32)) * $(512)) + (((r485_z / $(192)) % $(8)) * $(64)) + ((r485_z % $(192)) - $(128)) + $(16384)) / $(64)) % $(8)) * $(18432)) + (((((((r485_z / $(1536)) % $(32)) * $(512)) + (((r485_z / $(192)) % $(8)) * $(64)) + ((r485_z % $(192)) - $(128)) + $(16384)) / $(512)) % $(288)) * $(64)) + (((((r485_z / $(1536)) % $(32)) * $(512)) + (((r485_z / $(192)) % $(8)) * $(64)) + ((r485_z % $(192)) - $(128)) + $(16384)) % $(64))), dtype=tl.bfloat16))
    r485_value = r485_value.to(tl.bfloat16)
    r485_value = tl.broadcast_to(r485_value, [$(49152)])
    tl.store($(candidate_vcat_82_ptr) + r485_z, r485_value)
    r486_z = tl.arange($(147456))
    r486_value = tl.where((r486_z % $(192)) < $(128), tl.load($(candidate_vbmm_118_ptr) + ((((((((r486_z / $(1536)) % $(96)) * $(1024)) + (((r486_z / $(192)) % $(8)) * $(128)) + ((r486_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r486_z / $(1536)) % $(96)) * $(1024)) + (((r486_z / $(192)) % $(8)) * $(128)) + ((r486_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r486_z / $(1536)) % $(96)) * $(1024)) + (((r486_z / $(192)) % $(8)) * $(128)) + ((r486_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r486_z / $(1536)) % $(96)) * $(576)) + ((((r486_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r486_value = r486_value.to(tl.bfloat16)
    r486_value = tl.broadcast_to(r486_value, [$(147456)])
    tl.store($(candidate_vcat_83_ptr) + r486_z, r486_value)
    r487_z = tl.arange($(49152))
    r487_value = tl.load($(candidate_vcat_82_ptr) + ((((r487_z / $(192)) % $(32)) * $(1536)) + (((r487_z / $(6144)) % $(8)) * $(192)) + (r487_z % $(192))), dtype=tl.bfloat16)
    r487_value = r487_value.to(tl.float32)
    r487_value = tl.broadcast_to(r487_value, [$(49152)])
    tl.store($(candidate_vto_copy_144_ptr) + r487_z, r487_value)
    r488_z = tl.arange($(98304))
    r488_value = tl.load($(candidate_vcat_83_ptr) + ((((r488_z / $(192)) % $(64)) * $(1536)) + (((r488_z / $(12288)) % $(8)) * $(192)) + (r488_z % $(192))), dtype=tl.bfloat16)
    r488_value = r488_value.to(tl.float32)
    r488_value = tl.broadcast_to(r488_value, [$(98304)])
    tl.store($(candidate_vto_copy_145_ptr) + r488_z, r488_value)
    r489_row = tl.arange($(32))
    r489_col = tl.arange($(64))
    r489_lane = tl.arange($(192))
    for r489_group in range($(8)) {
      r489_left = tl.load($(candidate_vto_copy_144_ptr) + (((((((r489_group % $(8)) * $(6144)) + (r489_row[:, None] * $(192)) + r489_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r489_group % $(8)) * $(6144)) + (r489_row[:, None] * $(192)) + r489_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r489_group % $(8)) * $(6144)) + (r489_row[:, None] * $(192)) + r489_lane[None, :]) % $(192))), dtype=tl.float32)
      r489_right = tl.load($(candidate_vto_copy_145_ptr) + (((((((r489_group % $(8)) * $(12288)) + (r489_lane[:, None] * $(64)) + r489_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r489_group % $(8)) * $(12288)) + (r489_lane[:, None] * $(64)) + r489_col[None, :]) % $(64)) * $(192)) + (((((r489_group % $(8)) * $(12288)) + (r489_lane[:, None] * $(64)) + r489_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r489_value = tl.dot(r489_left, r489_right)
      r489_value = r489_value.to(tl.float32)
      tl.store($(candidate_vbmm_120_ptr) + ((r489_group * $(2048)) + (r489_row[:, None] * $(64)) + r489_col[None, :]), r489_value)
    }
    r490_z = tl.arange($(16384))
    r490_value = (tl.load($(candidate_vbmm_120_ptr) + ((((r490_z / $(2048)) % $(8)) * $(2048)) + (((r490_z / $(64)) % $(32)) * $(64)) + (r490_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r490_value = r490_value.to(tl.float32)
    r490_value = tl.broadcast_to(r490_value, [$(16384)])
    tl.store($(candidate_vmul_72_ptr) + r490_z, r490_value)
    r491_z = tl.arange($(32))
    r491_value = (tl.cast(r491_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r491_value = tl.broadcast_to(r491_value, [$(32)])
    tl.store($(candidate_vadd_69_ptr) + r491_z, r491_value)
    r492_z = tl.arange($(32))
    r492_value = (tl.load($(candidate_vadd_69_ptr) + r492_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r492_value = tl.broadcast_to(r492_value, [$(32)])
    tl.store($(candidate_vsub_52_ptr) + r492_z, r492_value)
    r493_z = tl.arange($(2048))
    r493_value = (tl.cast((r493_z % $(64)), tl.int64) > tl.load($(candidate_vsub_52_ptr) + ((r493_z / $(64)) % $(32)), dtype=tl.int64))
    r493_value = tl.broadcast_to(r493_value, [$(2048)])
    tl.store($(candidate_vgt_23_ptr) + r493_z, r493_value)
    r494_z = tl.arange($(16384))
    r494_value = tl.where(tl.load($(candidate_vgt_23_ptr) + ((((r494_z / $(64)) % $(32)) * $(64)) + (r494_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_72_ptr) + ((((r494_z / $(2048)) % $(8)) * $(2048)) + (((r494_z / $(64)) % $(32)) * $(64)) + (r494_z % $(64))), dtype=tl.float32))
    r494_value = r494_value.to(tl.float32)
    r494_value = tl.broadcast_to(r494_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_23_ptr) + r494_z, r494_value)
    r495_z = tl.arange($(16384))
    r495_lane = tl.arange($(64))
    r495_row_values = tl.load($(candidate_vmasked_fill_23_ptr) + ((((r495_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r495_z[:, None] / $(64)) % $(32)) * $(64)) + (r495_lane[None, :] % $(64))), dtype=tl.float32)
    r495_maximum = tl.max(r495_row_values, axis=1, keep_dims=true)
    r495_weights = tl.exp(r495_row_values - r495_maximum)
    r495_total = tl.sum(r495_weights, axis=1)
    r495_maximum = tl.max(r495_row_values, axis=1)
    r495_value = tl.exp(tl.load($(candidate_vmasked_fill_23_ptr) + ((((r495_z / $(2048)) % $(8)) * $(2048)) + (((r495_z / $(64)) % $(32)) * $(64)) + (r495_z % $(64))), dtype=tl.float32) - r495_maximum) / r495_total
    r495_value = r495_value.to(tl.float32)
    r495_value = tl.broadcast_to(r495_value, [$(16384)])
    tl.store($(candidate_vsoftmax_16_ptr) + r495_z, r495_value)
    r496_z = tl.arange($(16384))
    r496_value = tl.load($(candidate_vsoftmax_16_ptr) + ((((r496_z / $(2048)) % $(8)) * $(2048)) + (((r496_z / $(64)) % $(32)) * $(64)) + (r496_z % $(64))), dtype=tl.float32)
    r496_value = r496_value.to(tl.bfloat16)
    r496_value = tl.broadcast_to(r496_value, [$(16384)])
    tl.store($(candidate_vto_copy_146_ptr) + r496_z, r496_value)
    r497_row = tl.arange($(32))
    r497_col = tl.arange($(128))
    r497_lane = tl.arange($(64))
    for r497_group in range($(8)) {
      r497_left = tl.load($(candidate_vto_copy_146_ptr) + (((((((r497_group % $(8)) * $(2048)) + (r497_row[:, None] * $(64)) + r497_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r497_group % $(8)) * $(2048)) + (r497_row[:, None] * $(64)) + r497_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r497_group % $(8)) * $(2048)) + (r497_row[:, None] * $(64)) + r497_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r497_right = tl.load($(candidate_vbmm_119_ptr) + (((((((r497_group % $(8)) * $(8192)) + (r497_lane[:, None] * $(128)) + r497_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r497_group % $(8)) * $(8192)) + (r497_lane[:, None] * $(128)) + r497_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r497_group % $(8)) * $(8192)) + (r497_lane[:, None] * $(128)) + r497_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r497_value = tl.dot(r497_left, r497_right)
      r497_value = r497_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_121_ptr) + ((r497_group * $(4096)) + (r497_row[:, None] * $(128)) + r497_col[None, :]), r497_value)
    }
    r498_z = tl.arange($(256))
    r498_lane = tl.arange($(64))
    r498_row_values = tl.load($(candidate_vmasked_fill_23_ptr) + ((((r498_z[:, None] / $(32)) % $(8)) * $(2048)) + ((r498_z[:, None] % $(32)) * $(64)) + (r498_lane[None, :] % $(64))), dtype=tl.float32)
    r498_maximum = tl.max(r498_row_values, axis=1, keep_dims=true)
    r498_weights = tl.exp(r498_row_values - r498_maximum)
    r498_total = tl.sum(r498_weights, axis=1)
    r498_maximum = tl.max(r498_row_values, axis=1)
    r498_value = r498_maximum + tl.log(r498_total)
    r498_value = r498_value.to(tl.float32)
    r498_value = tl.broadcast_to(r498_value, [$(256)])
    tl.store($(candidate_vlogsumexp_16_ptr) + r498_z, r498_value)
    r499_z = tl.arange($(32768))
    r499_value = tl.load($(candidate_vbmm_121_ptr) + ((((((((r499_z / $(128)) % $(8)) * $(4096)) + ((((r499_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r499_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r499_z / $(128)) % $(8)) * $(32)) + (((r499_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r499_z % $(128))), dtype=tl.bfloat16)
    r499_value = r499_value.to(tl.bfloat16)
    r499_value = tl.broadcast_to(r499_value, [$(32768)])
    tl.store($(candidate_vcat_84_ptr) + r499_z, r499_value)
    r500_z = tl.arange($(256))
    r500_value = tl.load($(candidate_vlogsumexp_16_ptr) + (((r500_z % $(8)) * $(32)) + ((((r500_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r500_value = r500_value.to(tl.float32)
    r500_value = tl.broadcast_to(r500_value, [$(256)])
    tl.store($(candidate_vcat_85_ptr) + r500_z, r500_value)
    r501_z = tl.arange($(49152))
    r501_value = tl.load($(candidate_vcat_82_ptr) + ((((r501_z / $(192)) % $(32)) * $(1536)) + (((r501_z / $(6144)) % $(8)) * $(192)) + (r501_z % $(192))), dtype=tl.bfloat16)
    r501_value = r501_value.to(tl.float32)
    r501_value = tl.broadcast_to(r501_value, [$(49152)])
    tl.store($(candidate_vto_copy_147_ptr) + r501_z, r501_value)
    r502_z = tl.arange($(98304))
    r502_value = tl.load($(candidate_vcat_83_ptr) + ((((r502_z / $(192)) % $(64)) * $(1536)) + (((r502_z / $(12288)) % $(8)) * $(192)) + (r502_z % $(192))), dtype=tl.bfloat16)
    r502_value = r502_value.to(tl.float32)
    r502_value = tl.broadcast_to(r502_value, [$(98304)])
    tl.store($(candidate_vto_copy_148_ptr) + r502_z, r502_value)
    r503_z = tl.arange($(65536))
    r503_value = tl.load($(candidate_vbmm_119_ptr) + ((((r503_z / $(128)) % $(64)) * $(1024)) + (((r503_z / $(8192)) % $(8)) * $(128)) + (r503_z % $(128))), dtype=tl.bfloat16)
    r503_value = r503_value.to(tl.float32)
    r503_value = tl.broadcast_to(r503_value, [$(65536)])
    tl.store($(candidate_vto_copy_149_ptr) + r503_z, r503_value)
    r504_z = tl.arange($(32768))
    r504_value = tl.load($(candidate_vmm_22_ptr) + ((((r504_z / $(128)) % $(32)) * $(1024)) + (((r504_z / $(4096)) % $(8)) * $(128)) + (r504_z % $(128)) + $(32768)), dtype=tl.bfloat16)
    r504_value = r504_value.to(tl.float32)
    r504_value = tl.broadcast_to(r504_value, [$(32768)])
    tl.store($(candidate_vto_copy_150_ptr) + r504_z, r504_value)
    r505_z = tl.arange($(32768))
    r505_value = tl.load($(candidate_vcat_84_ptr) + ((((r505_z / $(128)) % $(32)) * $(1024)) + (((r505_z / $(4096)) % $(8)) * $(128)) + (r505_z % $(128))), dtype=tl.bfloat16)
    r505_value = r505_value.to(tl.float32)
    r505_value = tl.broadcast_to(r505_value, [$(32768)])
    tl.store($(candidate_vto_copy_151_ptr) + r505_z, r505_value)
    r506_row = tl.arange($(32))
    r506_col = tl.arange($(64))
    r506_lane = tl.arange($(192))
    for r506_group in range($(8)) {
      r506_left = tl.load($(candidate_vto_copy_147_ptr) + (((((((r506_group % $(8)) * $(6144)) + (r506_row[:, None] * $(192)) + r506_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r506_group % $(8)) * $(6144)) + (r506_row[:, None] * $(192)) + r506_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r506_group % $(8)) * $(6144)) + (r506_row[:, None] * $(192)) + r506_lane[None, :]) % $(192))), dtype=tl.float32)
      r506_right = tl.load($(candidate_vto_copy_148_ptr) + (((((((r506_group % $(8)) * $(12288)) + (r506_lane[:, None] * $(64)) + r506_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r506_group % $(8)) * $(12288)) + (r506_lane[:, None] * $(64)) + r506_col[None, :]) % $(64)) * $(192)) + (((((r506_group % $(8)) * $(12288)) + (r506_lane[:, None] * $(64)) + r506_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r506_value = tl.dot(r506_left, r506_right)
      r506_value = r506_value.to(tl.float32)
      tl.store($(candidate_vbmm_122_ptr) + ((r506_group * $(2048)) + (r506_row[:, None] * $(64)) + r506_col[None, :]), r506_value)
    }
    r507_z = tl.arange($(16384))
    r507_value = (tl.load($(candidate_vbmm_122_ptr) + ((((r507_z / $(2048)) % $(8)) * $(2048)) + (((r507_z / $(64)) % $(32)) * $(64)) + (r507_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r507_value = r507_value.to(tl.float32)
    r507_value = tl.broadcast_to(r507_value, [$(16384)])
    tl.store($(candidate_vmul_73_ptr) + r507_z, r507_value)
    r508_z = tl.arange($(32))
    r508_value = (tl.cast(r508_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r508_value = tl.broadcast_to(r508_value, [$(32)])
    tl.store($(candidate_vadd_70_ptr) + r508_z, r508_value)
    r509_z = tl.arange($(32))
    r509_value = (tl.load($(candidate_vadd_70_ptr) + r509_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r509_value = tl.broadcast_to(r509_value, [$(32)])
    tl.store($(candidate_vsub_53_ptr) + r509_z, r509_value)
    r510_z = tl.arange($(2048))
    r510_value = (tl.cast((r510_z % $(64)), tl.int64) > tl.load($(candidate_vsub_53_ptr) + ((r510_z / $(64)) % $(32)), dtype=tl.int64))
    r510_value = tl.broadcast_to(r510_value, [$(2048)])
    tl.store($(candidate_vgt_24_ptr) + r510_z, r510_value)
    r511_z = tl.arange($(16384))
    r511_value = tl.where(tl.load($(candidate_vgt_24_ptr) + ((((r511_z / $(64)) % $(32)) * $(64)) + (r511_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_73_ptr) + ((((r511_z / $(2048)) % $(8)) * $(2048)) + (((r511_z / $(64)) % $(32)) * $(64)) + (r511_z % $(64))), dtype=tl.float32))
    r511_value = r511_value.to(tl.float32)
    r511_value = tl.broadcast_to(r511_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_24_ptr) + r511_z, r511_value)
    r512_z = tl.arange($(16384))
    r512_value = (tl.load($(candidate_vmasked_fill_24_ptr) + ((((r512_z / $(2048)) % $(8)) * $(2048)) + (((r512_z / $(64)) % $(32)) * $(64)) + (r512_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_85_ptr) + ((((r512_z / $(64)) % $(32)) * $(8)) + ((r512_z / $(2048)) % $(8))), dtype=tl.float32))
    r512_value = r512_value.to(tl.float32)
    r512_value = tl.broadcast_to(r512_value, [$(16384)])
    tl.store($(candidate_vsub_54_ptr) + r512_z, r512_value)
    r513_z = tl.arange($(16384))
    r513_value = tl.exp(tl.load($(candidate_vsub_54_ptr) + ((((r513_z / $(2048)) % $(8)) * $(2048)) + (((r513_z / $(64)) % $(32)) * $(64)) + (r513_z % $(64))), dtype=tl.float32))
    r513_value = r513_value.to(tl.float32)
    r513_value = tl.broadcast_to(r513_value, [$(16384)])
    tl.store($(candidate_vexp_7_ptr) + r513_z, r513_value)
    r514_z = tl.arange($(32768))
    r514_value = (tl.load($(candidate_vto_copy_150_ptr) + ((((r514_z / $(4096)) % $(8)) * $(4096)) + (((r514_z / $(128)) % $(32)) * $(128)) + (r514_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_151_ptr) + ((((r514_z / $(4096)) % $(8)) * $(4096)) + (((r514_z / $(128)) % $(32)) * $(128)) + (r514_z % $(128))), dtype=tl.float32))
    r514_value = r514_value.to(tl.float32)
    r514_value = tl.broadcast_to(r514_value, [$(32768)])
    tl.store($(candidate_vmul_74_ptr) + r514_z, r514_value)
    r515_z = tl.arange($(256))
    r515_lane = tl.arange($(128))
    r515_value = tl.sum(tl.load($(candidate_vmul_74_ptr) + ((((r515_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r515_z[:, None] % $(32)) * $(128)) + r515_lane[None, :]), dtype=tl.float32), axis=1)
    r515_value = r515_value.to(tl.float32)
    r515_value = tl.broadcast_to(r515_value, [$(256)])
    tl.store($(candidate_vsum_15_ptr) + r515_z, r515_value)
    r516_row = tl.arange($(32))
    r516_col = tl.arange($(64))
    r516_lane = tl.arange($(128))
    for r516_group in range($(8)) {
      r516_left = tl.load($(candidate_vto_copy_150_ptr) + (((((((r516_group % $(8)) * $(4096)) + (r516_row[:, None] * $(128)) + r516_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r516_group % $(8)) * $(4096)) + (r516_row[:, None] * $(128)) + r516_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r516_group % $(8)) * $(4096)) + (r516_row[:, None] * $(128)) + r516_lane[None, :]) % $(128))), dtype=tl.float32)
      r516_right = tl.load($(candidate_vto_copy_149_ptr) + (((((((r516_group % $(8)) * $(8192)) + (r516_lane[:, None] * $(64)) + r516_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r516_group % $(8)) * $(8192)) + (r516_lane[:, None] * $(64)) + r516_col[None, :]) % $(64)) * $(128)) + (((((r516_group % $(8)) * $(8192)) + (r516_lane[:, None] * $(64)) + r516_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r516_value = tl.dot(r516_left, r516_right)
      r516_value = r516_value.to(tl.float32)
      tl.store($(candidate_vbmm_123_ptr) + ((r516_group * $(2048)) + (r516_row[:, None] * $(64)) + r516_col[None, :]), r516_value)
    }
    r517_z = tl.arange($(16384))
    r517_value = (tl.load($(candidate_vbmm_123_ptr) + ((((r517_z / $(2048)) % $(8)) * $(2048)) + (((r517_z / $(64)) % $(32)) * $(64)) + (r517_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_15_ptr) + ((((r517_z / $(2048)) % $(8)) * $(32)) + ((r517_z / $(64)) % $(32))), dtype=tl.float32))
    r517_value = r517_value.to(tl.float32)
    r517_value = tl.broadcast_to(r517_value, [$(16384)])
    tl.store($(candidate_vsub_55_ptr) + r517_z, r517_value)
    r518_z = tl.arange($(16384))
    r518_value = (tl.load($(candidate_vexp_7_ptr) + ((((r518_z / $(2048)) % $(8)) * $(2048)) + (((r518_z / $(64)) % $(32)) * $(64)) + (r518_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_55_ptr) + ((((r518_z / $(2048)) % $(8)) * $(2048)) + (((r518_z / $(64)) % $(32)) * $(64)) + (r518_z % $(64))), dtype=tl.float32))
    r518_value = r518_value.to(tl.float32)
    r518_value = tl.broadcast_to(r518_value, [$(16384)])
    tl.store($(candidate_vmul_75_ptr) + r518_z, r518_value)
    r519_row = tl.arange($(32))
    r519_col = tl.arange($(192))
    r519_lane = tl.arange($(64))
    for r519_group in range($(8)) {
      r519_left = tl.load($(candidate_vmul_75_ptr) + (((((((r519_group % $(8)) * $(2048)) + (r519_row[:, None] * $(64)) + r519_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r519_group % $(8)) * $(2048)) + (r519_row[:, None] * $(64)) + r519_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r519_group % $(8)) * $(2048)) + (r519_row[:, None] * $(64)) + r519_lane[None, :]) % $(64))), dtype=tl.float32)
      r519_right = tl.load($(candidate_vto_copy_148_ptr) + (((((((r519_group % $(8)) * $(12288)) + (r519_lane[:, None] * $(192)) + r519_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r519_group % $(8)) * $(12288)) + (r519_lane[:, None] * $(192)) + r519_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r519_group % $(8)) * $(12288)) + (r519_lane[:, None] * $(192)) + r519_col[None, :]) % $(192))), dtype=tl.float32)
      r519_value = tl.dot(r519_left, r519_right)
      r519_value = r519_value.to(tl.float32)
      tl.store($(candidate_vbmm_124_ptr) + ((r519_group * $(6144)) + (r519_row[:, None] * $(192)) + r519_col[None, :]), r519_value)
    }
    r520_z = tl.arange($(49152))
    r520_value = (tl.load($(candidate_vbmm_124_ptr) + ((((r520_z / $(6144)) % $(8)) * $(6144)) + (((r520_z / $(192)) % $(32)) * $(192)) + (r520_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r520_value = r520_value.to(tl.float32)
    r520_value = tl.broadcast_to(r520_value, [$(49152)])
    tl.store($(candidate_vmul_76_ptr) + r520_z, r520_value)
    r521_z = tl.arange($(49152))
    r521_value = tl.load($(candidate_vmul_76_ptr) + ((((r521_z / $(192)) % $(8)) * $(6144)) + (((r521_z / $(1536)) % $(32)) * $(192)) + (r521_z % $(192))), dtype=tl.float32)
    r521_value = r521_value.to(tl.bfloat16)
    r521_value = tl.broadcast_to(r521_value, [$(49152)])
    tl.store($(candidate_vto_copy_152_ptr) + r521_z, r521_value)
    r522_row = tl.arange($(64))
    r522_col = tl.arange($(192))
    r522_lane = tl.arange($(32))
    for r522_group in range($(8)) {
      r522_left = tl.load($(candidate_vmul_75_ptr) + (((((((r522_group % $(8)) * $(2048)) + (r522_row[:, None] * $(32)) + r522_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r522_group % $(8)) * $(2048)) + (r522_row[:, None] * $(32)) + r522_lane[None, :]) % $(32)) * $(64)) + (((((r522_group % $(8)) * $(2048)) + (r522_row[:, None] * $(32)) + r522_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r522_right = tl.load($(candidate_vto_copy_147_ptr) + (((((((r522_group % $(8)) * $(6144)) + (r522_lane[:, None] * $(192)) + r522_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r522_group % $(8)) * $(6144)) + (r522_lane[:, None] * $(192)) + r522_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r522_group % $(8)) * $(6144)) + (r522_lane[:, None] * $(192)) + r522_col[None, :]) % $(192))), dtype=tl.float32)
      r522_value = tl.dot(r522_left, r522_right)
      r522_value = r522_value.to(tl.float32)
      tl.store($(candidate_vbmm_125_ptr) + ((r522_group * $(12288)) + (r522_row[:, None] * $(192)) + r522_col[None, :]), r522_value)
    }
    r523_z = tl.arange($(98304))
    r523_value = (tl.load($(candidate_vbmm_125_ptr) + ((((r523_z / $(12288)) % $(8)) * $(12288)) + (((r523_z / $(192)) % $(64)) * $(192)) + (r523_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r523_value = r523_value.to(tl.float32)
    r523_value = tl.broadcast_to(r523_value, [$(98304)])
    tl.store($(candidate_vmul_77_ptr) + r523_z, r523_value)
    r524_z = tl.arange($(98304))
    r524_value = tl.load($(candidate_vmul_77_ptr) + ((((r524_z / $(192)) % $(8)) * $(12288)) + (((r524_z / $(1536)) % $(64)) * $(192)) + (r524_z % $(192))), dtype=tl.float32)
    r524_value = r524_value.to(tl.bfloat16)
    r524_value = tl.broadcast_to(r524_value, [$(98304)])
    tl.store($(candidate_vto_copy_153_ptr) + r524_z, r524_value)
    r525_row = tl.arange($(64))
    r525_col = tl.arange($(128))
    r525_lane = tl.arange($(32))
    for r525_group in range($(8)) {
      r525_left = tl.load($(candidate_vexp_7_ptr) + (((((((r525_group % $(8)) * $(2048)) + (r525_row[:, None] * $(32)) + r525_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r525_group % $(8)) * $(2048)) + (r525_row[:, None] * $(32)) + r525_lane[None, :]) % $(32)) * $(64)) + (((((r525_group % $(8)) * $(2048)) + (r525_row[:, None] * $(32)) + r525_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r525_right = tl.load($(candidate_vto_copy_150_ptr) + (((((((r525_group % $(8)) * $(4096)) + (r525_lane[:, None] * $(128)) + r525_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r525_group % $(8)) * $(4096)) + (r525_lane[:, None] * $(128)) + r525_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r525_group % $(8)) * $(4096)) + (r525_lane[:, None] * $(128)) + r525_col[None, :]) % $(128))), dtype=tl.float32)
      r525_value = tl.dot(r525_left, r525_right)
      r525_value = r525_value.to(tl.float32)
      tl.store($(candidate_vbmm_126_ptr) + ((r525_group * $(8192)) + (r525_row[:, None] * $(128)) + r525_col[None, :]), r525_value)
    }
    r526_z = tl.arange($(65536))
    r526_value = tl.load($(candidate_vbmm_126_ptr) + ((((r526_z / $(128)) % $(8)) * $(8192)) + (((r526_z / $(1024)) % $(64)) * $(128)) + (r526_z % $(128))), dtype=tl.float32)
    r526_value = r526_value.to(tl.bfloat16)
    r526_value = tl.broadcast_to(r526_value, [$(65536)])
    tl.store($(candidate_vto_copy_154_ptr) + r526_z, r526_value)
    r527_z = tl.arange($(147456))
    r527_value = tl.where(((r527_z / $(1536)) % $(96)) < $(64), tl.load($(candidate_vto_copy_153_ptr) + ((((((r527_z / $(1536)) % $(96)) - $(0)) % $(64)) * $(1536)) + (((r527_z / $(192)) % $(8)) * $(192)) + (r527_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r527_value = r527_value.to(tl.bfloat16)
    r527_value = tl.broadcast_to(r527_value, [$(147456)])
    tl.store($(candidate_vcat_86_ptr) + r527_z, r527_value)
    r528_z = tl.arange($(98304))
    r528_value = tl.where(((r528_z / $(1024)) % $(96)) < $(64), tl.load($(candidate_vto_copy_154_ptr) + ((((((r528_z / $(1024)) % $(96)) - $(0)) % $(64)) * $(1024)) + (((r528_z / $(128)) % $(8)) * $(128)) + (r528_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r528_value = r528_value.to(tl.bfloat16)
    r528_value = tl.broadcast_to(r528_value, [$(98304)])
    tl.store($(candidate_vcat_87_ptr) + r528_z, r528_value)
    r529_z = tl.arange($(49152))
    r529_value = tl.load($(candidate_vto_copy_152_ptr) + ((((((r529_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(1536)) + (((r529_z / $(192)) % $(8)) * $(192)) + (r529_z % $(192))), dtype=tl.bfloat16)
    r529_value = r529_value.to(tl.bfloat16)
    r529_value = tl.broadcast_to(r529_value, [$(49152)])
    tl.store($(candidate_vcat_88_ptr) + r529_z, r529_value)
    r530_z = tl.arange($(147456))
    r530_value = tl.load($(candidate_vcat_86_ptr) + ((((((r530_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r530_z / $(192)) % $(8)) * $(192)) + (r530_z % $(192))), dtype=tl.bfloat16)
    r530_value = r530_value.to(tl.bfloat16)
    r530_value = tl.broadcast_to(r530_value, [$(147456)])
    tl.store($(candidate_vcat_89_ptr) + r530_z, r530_value)
    r531_z = tl.arange($(98304))
    r531_value = tl.load($(candidate_vcat_87_ptr) + ((((((r531_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r531_z / $(128)) % $(8)) * $(128)) + (r531_z % $(128))), dtype=tl.bfloat16)
    r531_value = r531_value.to(tl.bfloat16)
    r531_value = tl.broadcast_to(r531_value, [$(98304)])
    tl.store($(candidate_vcat_90_ptr) + r531_z, r531_value)
    r532_z = tl.arange($(6144))
    r532_lane = tl.arange($(8))
    r532_value = tl.sum(tl.load($(candidate_vcat_89_ptr) + ((((r532_z[:, None] / $(64)) % $(96)) * $(1536)) + (r532_lane[None, :] * $(192)) + (r532_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r532_value = r532_value.to(tl.bfloat16)
    r532_value = tl.broadcast_to(r532_value, [$(6144)])
    tl.store($(candidate_vsum_16_ptr) + r532_z, r532_value)
    r533_row = tl.arange($(96))
    r533_col = tl.arange($(512))
    r533_lane = tl.arange($(1024))
    for r533_group in range($(1)) {
      r533_left = tl.load($(candidate_vcat_89_ptr) + ((((((r533_row[:, None] * $(1024)) + r533_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + ((((r533_row[:, None] * $(1024)) + r533_lane[None, :]) % $(8)) * $(192)) + ((((r533_row[:, None] * $(1024)) + r533_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r533_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r533_lane[:, None] * $(512)) + r533_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r533_lane[:, None] * $(512)) + r533_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r533_lane[:, None] * $(512)) + r533_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r533_value = tl.dot(r533_left, r533_right)
      r533_value = r533_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_127_ptr) + ((r533_group * $(49152)) + (r533_row[:, None] * $(512)) + r533_col[None, :]), r533_value)
    }
    r534_row = tl.arange($(96))
    r534_col = tl.arange($(512))
    r534_lane = tl.arange($(1024))
    for r534_group in range($(1)) {
      r534_left = tl.load($(candidate_vcat_90_ptr) + ((((((r534_row[:, None] * $(1024)) + r534_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + ((((r534_row[:, None] * $(1024)) + r534_lane[None, :]) % $(8)) * $(128)) + ((((r534_row[:, None] * $(1024)) + r534_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r534_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r534_lane[:, None] * $(512)) + r534_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r534_lane[:, None] * $(512)) + r534_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r534_lane[:, None] * $(512)) + r534_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r534_value = tl.dot(r534_left, r534_right)
      r534_value = r534_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_128_ptr) + ((r534_group * $(49152)) + (r534_row[:, None] * $(512)) + r534_col[None, :]), r534_value)
    }
    r535_z = tl.arange($(49152))
    r535_value = (tl.load($(candidate_vbmm_127_ptr) + ((((r535_z / $(512)) % $(96)) * $(512)) + (r535_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_128_ptr) + ((((r535_z / $(512)) % $(96)) * $(512)) + (r535_z % $(512))), dtype=tl.bfloat16))
    r535_value = r535_value.to(tl.bfloat16)
    r535_value = tl.broadcast_to(r535_value, [$(49152)])
    tl.store($(candidate_vadd_71_ptr) + r535_z, r535_value)
    r536_z = tl.arange($(147456))
    r536_value = tl.where(((((r536_z / $(512)) % $(288)) >= $(0)) & (((r536_z / $(512)) % $(288)) < $(96)) & ((((r536_z / $(512)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_71_ptr) + ((((((r536_z / $(512)) % $(288)) - $(0)) % $(96)) * $(512)) + (r536_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r536_value = r536_value.to(tl.bfloat16)
    r536_value = tl.broadcast_to(r536_value, [$(147456)])
    tl.store($(candidate_vslice_backward_28_ptr) + r536_z, r536_value)
    r537_z = tl.arange($(147456))
    r537_value = (tl.load($(candidate_vadd_65_ptr) + ((((r537_z / $(512)) % $(288)) * $(512)) + (r537_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_28_ptr) + ((((r537_z / $(512)) % $(288)) * $(512)) + (r537_z % $(512))), dtype=tl.bfloat16))
    r537_value = r537_value.to(tl.bfloat16)
    r537_value = tl.broadcast_to(r537_value, [$(147456)])
    tl.store($(candidate_vadd_72_ptr) + r537_z, r537_value)
    r538_z = tl.arange($(18432))
    r538_value = tl.where(((((r538_z / $(64)) % $(288)) >= $(0)) & (((r538_z / $(64)) % $(288)) < $(96)) & ((((r538_z / $(64)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_16_ptr) + ((((((r538_z / $(64)) % $(288)) - $(0)) % $(96)) * $(64)) + (r538_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r538_value = r538_value.to(tl.bfloat16)
    r538_value = tl.broadcast_to(r538_value, [$(18432)])
    tl.store($(candidate_vslice_backward_29_ptr) + r538_z, r538_value)
    r539_z = tl.arange($(18432))
    r539_value = (tl.load($(candidate_vadd_66_ptr) + ((((r539_z / $(64)) % $(288)) * $(64)) + (r539_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_29_ptr) + ((((r539_z / $(64)) % $(288)) * $(64)) + (r539_z % $(64))), dtype=tl.bfloat16))
    r539_value = r539_value.to(tl.bfloat16)
    r539_value = tl.broadcast_to(r539_value, [$(18432)])
    tl.store($(candidate_vadd_73_ptr) + r539_z, r539_value)
    r540_z = tl.arange($(294912))
    r540_value = tl.where(((((r540_z / $(1024)) % $(288)) >= $(32)) & (((r540_z / $(1024)) % $(288)) < $(64)) & ((((r540_z / $(1024)) % $(288)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vcat_88_ptr) + ((((((r540_z / $(1024)) % $(288)) - $(32)) % $(32)) * $(1536)) + (((r540_z / $(128)) % $(8)) * $(192)) + (r540_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r540_value = r540_value.to(tl.bfloat16)
    r540_value = tl.broadcast_to(r540_value, [$(294912)])
    tl.store($(candidate_vslice_backward_30_ptr) + r540_z, r540_value)
    r541_z = tl.arange($(294912))
    r541_value = (tl.load($(candidate_vadd_67_ptr) + ((((r541_z / $(1024)) % $(288)) * $(1024)) + (((r541_z / $(128)) % $(8)) * $(128)) + (r541_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_30_ptr) + ((((r541_z / $(1024)) % $(288)) * $(1024)) + (((r541_z / $(128)) % $(8)) * $(128)) + (r541_z % $(128))), dtype=tl.bfloat16))
    r541_value = r541_value.to(tl.bfloat16)
    r541_value = tl.broadcast_to(r541_value, [$(294912)])
    tl.store($(candidate_vadd_74_ptr) + r541_z, r541_value)
    r542_z = tl.arange($(147456))
    r542_value = tl.where(((((r542_z / $(512)) % $(288)) >= $(32)) & (((r542_z / $(512)) % $(288)) < $(64)) & ((((r542_z / $(512)) % $(288)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vcat_88_ptr) + ((((((r542_z / $(512)) % $(288)) - $(32)) % $(32)) * $(1536)) + (((r542_z / $(64)) % $(8)) * $(192)) + (r542_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r542_value = r542_value.to(tl.bfloat16)
    r542_value = tl.broadcast_to(r542_value, [$(147456)])
    tl.store($(candidate_vslice_backward_31_ptr) + r542_z, r542_value)
    r543_z = tl.arange($(147456))
    r543_value = (tl.load($(candidate_vadd_68_ptr) + ((((r543_z / $(512)) % $(288)) * $(512)) + (((r543_z / $(64)) % $(8)) * $(64)) + (r543_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_31_ptr) + ((((r543_z / $(512)) % $(288)) * $(512)) + (((r543_z / $(64)) % $(8)) * $(64)) + (r543_z % $(64))), dtype=tl.bfloat16))
    r543_value = r543_value.to(tl.bfloat16)
    r543_value = tl.broadcast_to(r543_value, [$(147456)])
    tl.store($(candidate_vadd_75_ptr) + r543_z, r543_value)
    r544_row = tl.arange($(96))
    r544_col = tl.arange($(1024))
    r544_lane = tl.arange($(512))
    for r544_group in range($(1)) {
      r544_left = tl.load($(candidate_vcat_7_ptr) + ((((((r544_row[:, None] * $(512)) + r544_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r544_row[:, None] * $(512)) + r544_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r544_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r544_lane[:, None] * $(1024)) + r544_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r544_lane[:, None] * $(1024)) + r544_col[None, :]) % $(128)) * $(512)) + ((((r544_lane[:, None] * $(1024)) + r544_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r544_value = tl.dot(r544_left, r544_right)
      r544_value = r544_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_131_ptr) + ((r544_group * $(98304)) + (r544_row[:, None] * $(1024)) + r544_col[None, :]), r544_value)
    }
    r545_row = tl.arange($(96))
    r545_col = tl.arange($(1024))
    r545_lane = tl.arange($(512))
    for r545_group in range($(1)) {
      r545_left = tl.load($(candidate_vcat_7_ptr) + ((((((r545_row[:, None] * $(512)) + r545_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r545_row[:, None] * $(512)) + r545_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r545_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r545_lane[:, None] * $(1024)) + r545_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r545_lane[:, None] * $(1024)) + r545_col[None, :]) % $(128)) * $(512)) + ((((r545_lane[:, None] * $(1024)) + r545_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r545_value = tl.dot(r545_left, r545_right)
      r545_value = r545_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_132_ptr) + ((r545_group * $(98304)) + (r545_row[:, None] * $(1024)) + r545_col[None, :]), r545_value)
    }
    r546_z = tl.arange($(49152))
    r546_value = tl.where((r546_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r546_z / $(1536)) % $(32)) * $(1024)) + (((r546_z / $(192)) % $(8)) * $(128)) + ((r546_z % $(192)) - $(0))) / $(1024)) % $(288)) * $(1536)) + (((((((r546_z / $(1536)) % $(32)) * $(1024)) + (((r546_z / $(192)) % $(8)) * $(128)) + ((r546_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r546_z / $(1536)) % $(32)) * $(1024)) + (((r546_z / $(192)) % $(8)) * $(128)) + ((r546_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r546_z / $(1536)) % $(32)) * $(512)) + (((r546_z / $(192)) % $(8)) * $(64)) + ((r546_z % $(192)) - $(128))) / $(64)) % $(8)) * $(18432)) + (((((((r546_z / $(1536)) % $(32)) * $(512)) + (((r546_z / $(192)) % $(8)) * $(64)) + ((r546_z % $(192)) - $(128))) / $(512)) % $(288)) * $(64)) + (((((r546_z / $(1536)) % $(32)) * $(512)) + (((r546_z / $(192)) % $(8)) * $(64)) + ((r546_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r546_value = r546_value.to(tl.bfloat16)
    r546_value = tl.broadcast_to(r546_value, [$(49152)])
    tl.store($(candidate_vcat_92_ptr) + r546_z, r546_value)
    r547_z = tl.arange($(147456))
    r547_value = tl.where((r547_z % $(192)) < $(128), tl.load($(candidate_vbmm_131_ptr) + ((((((((r547_z / $(1536)) % $(96)) * $(1024)) + (((r547_z / $(192)) % $(8)) * $(128)) + ((r547_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r547_z / $(1536)) % $(96)) * $(1024)) + (((r547_z / $(192)) % $(8)) * $(128)) + ((r547_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r547_z / $(1536)) % $(96)) * $(1024)) + (((r547_z / $(192)) % $(8)) * $(128)) + ((r547_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r547_z / $(1536)) % $(96)) * $(576)) + ((((r547_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r547_value = r547_value.to(tl.bfloat16)
    r547_value = tl.broadcast_to(r547_value, [$(147456)])
    tl.store($(candidate_vcat_93_ptr) + r547_z, r547_value)
    r548_z = tl.arange($(49152))
    r548_value = tl.load($(candidate_vcat_92_ptr) + ((((r548_z / $(192)) % $(32)) * $(1536)) + (((r548_z / $(6144)) % $(8)) * $(192)) + (r548_z % $(192))), dtype=tl.bfloat16)
    r548_value = r548_value.to(tl.float32)
    r548_value = tl.broadcast_to(r548_value, [$(49152)])
    tl.store($(candidate_vto_copy_155_ptr) + r548_z, r548_value)
    r549_z = tl.arange($(49152))
    r549_value = tl.load($(candidate_vcat_93_ptr) + ((((r549_z / $(192)) % $(32)) * $(1536)) + (((r549_z / $(6144)) % $(8)) * $(192)) + (r549_z % $(192))), dtype=tl.bfloat16)
    r549_value = r549_value.to(tl.float32)
    r549_value = tl.broadcast_to(r549_value, [$(49152)])
    tl.store($(candidate_vto_copy_156_ptr) + r549_z, r549_value)
    r550_row = tl.arange($(32))
    r550_col = tl.arange($(32))
    r550_lane = tl.arange($(192))
    for r550_group in range($(8)) {
      r550_left = tl.load($(candidate_vto_copy_155_ptr) + (((((((r550_group % $(8)) * $(6144)) + (r550_row[:, None] * $(192)) + r550_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r550_group % $(8)) * $(6144)) + (r550_row[:, None] * $(192)) + r550_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r550_group % $(8)) * $(6144)) + (r550_row[:, None] * $(192)) + r550_lane[None, :]) % $(192))), dtype=tl.float32)
      r550_right = tl.load($(candidate_vto_copy_156_ptr) + (((((((r550_group % $(8)) * $(6144)) + (r550_lane[:, None] * $(32)) + r550_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r550_group % $(8)) * $(6144)) + (r550_lane[:, None] * $(32)) + r550_col[None, :]) % $(32)) * $(192)) + (((((r550_group % $(8)) * $(6144)) + (r550_lane[:, None] * $(32)) + r550_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r550_value = tl.dot(r550_left, r550_right)
      r550_value = r550_value.to(tl.float32)
      tl.store($(candidate_vbmm_133_ptr) + ((r550_group * $(1024)) + (r550_row[:, None] * $(32)) + r550_col[None, :]), r550_value)
    }
    r551_z = tl.arange($(8192))
    r551_value = (tl.load($(candidate_vbmm_133_ptr) + ((((r551_z / $(1024)) % $(8)) * $(1024)) + (((r551_z / $(32)) % $(32)) * $(32)) + (r551_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r551_value = r551_value.to(tl.float32)
    r551_value = tl.broadcast_to(r551_value, [$(8192)])
    tl.store($(candidate_vmul_78_ptr) + r551_z, r551_value)
    r552_z = tl.arange($(32))
    r552_value = (tl.cast(r552_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r552_value = tl.broadcast_to(r552_value, [$(32)])
    tl.store($(candidate_vadd_76_ptr) + r552_z, r552_value)
    r553_z = tl.arange($(32))
    r553_value = (tl.load($(candidate_vadd_76_ptr) + r553_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r553_value = tl.broadcast_to(r553_value, [$(32)])
    tl.store($(candidate_vsub_56_ptr) + r553_z, r553_value)
    r554_z = tl.arange($(1024))
    r554_value = (tl.cast((r554_z % $(32)), tl.int64) > tl.load($(candidate_vsub_56_ptr) + ((r554_z / $(32)) % $(32)), dtype=tl.int64))
    r554_value = tl.broadcast_to(r554_value, [$(1024)])
    tl.store($(candidate_vgt_25_ptr) + r554_z, r554_value)
    r555_z = tl.arange($(8192))
    r555_value = tl.where(tl.load($(candidate_vgt_25_ptr) + ((((r555_z / $(32)) % $(32)) * $(32)) + (r555_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_78_ptr) + ((((r555_z / $(1024)) % $(8)) * $(1024)) + (((r555_z / $(32)) % $(32)) * $(32)) + (r555_z % $(32))), dtype=tl.float32))
    r555_value = r555_value.to(tl.float32)
    r555_value = tl.broadcast_to(r555_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_25_ptr) + r555_z, r555_value)
    r556_z = tl.arange($(8192))
    r556_lane = tl.arange($(32))
    r556_row_values = tl.load($(candidate_vmasked_fill_25_ptr) + ((((r556_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r556_z[:, None] / $(32)) % $(32)) * $(32)) + (r556_lane[None, :] % $(32))), dtype=tl.float32)
    r556_maximum = tl.max(r556_row_values, axis=1, keep_dims=true)
    r556_weights = tl.exp(r556_row_values - r556_maximum)
    r556_total = tl.sum(r556_weights, axis=1)
    r556_maximum = tl.max(r556_row_values, axis=1)
    r556_value = tl.exp(tl.load($(candidate_vmasked_fill_25_ptr) + ((((r556_z / $(1024)) % $(8)) * $(1024)) + (((r556_z / $(32)) % $(32)) * $(32)) + (r556_z % $(32))), dtype=tl.float32) - r556_maximum) / r556_total
    r556_value = r556_value.to(tl.float32)
    r556_value = tl.broadcast_to(r556_value, [$(8192)])
    tl.store($(candidate_vsoftmax_17_ptr) + r556_z, r556_value)
    r557_z = tl.arange($(8192))
    r557_value = tl.load($(candidate_vsoftmax_17_ptr) + ((((r557_z / $(1024)) % $(8)) * $(1024)) + (((r557_z / $(32)) % $(32)) * $(32)) + (r557_z % $(32))), dtype=tl.float32)
    r557_value = r557_value.to(tl.bfloat16)
    r557_value = tl.broadcast_to(r557_value, [$(8192)])
    tl.store($(candidate_vto_copy_157_ptr) + r557_z, r557_value)
    r558_row = tl.arange($(32))
    r558_col = tl.arange($(128))
    r558_lane = tl.arange($(32))
    for r558_group in range($(8)) {
      r558_left = tl.load($(candidate_vto_copy_157_ptr) + (((((((r558_group % $(8)) * $(1024)) + (r558_row[:, None] * $(32)) + r558_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r558_group % $(8)) * $(1024)) + (r558_row[:, None] * $(32)) + r558_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r558_group % $(8)) * $(1024)) + (r558_row[:, None] * $(32)) + r558_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r558_right = tl.load($(candidate_vbmm_132_ptr) + (((((((r558_group % $(8)) * $(4096)) + (r558_lane[:, None] * $(128)) + r558_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r558_group % $(8)) * $(4096)) + (r558_lane[:, None] * $(128)) + r558_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r558_group % $(8)) * $(4096)) + (r558_lane[:, None] * $(128)) + r558_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r558_value = tl.dot(r558_left, r558_right)
      r558_value = r558_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_134_ptr) + ((r558_group * $(4096)) + (r558_row[:, None] * $(128)) + r558_col[None, :]), r558_value)
    }
    r559_z = tl.arange($(256))
    r559_lane = tl.arange($(32))
    r559_row_values = tl.load($(candidate_vmasked_fill_25_ptr) + ((((r559_z[:, None] / $(32)) % $(8)) * $(1024)) + ((r559_z[:, None] % $(32)) * $(32)) + (r559_lane[None, :] % $(32))), dtype=tl.float32)
    r559_maximum = tl.max(r559_row_values, axis=1, keep_dims=true)
    r559_weights = tl.exp(r559_row_values - r559_maximum)
    r559_total = tl.sum(r559_weights, axis=1)
    r559_maximum = tl.max(r559_row_values, axis=1)
    r559_value = r559_maximum + tl.log(r559_total)
    r559_value = r559_value.to(tl.float32)
    r559_value = tl.broadcast_to(r559_value, [$(256)])
    tl.store($(candidate_vlogsumexp_17_ptr) + r559_z, r559_value)
    r560_z = tl.arange($(32768))
    r560_value = tl.load($(candidate_vbmm_134_ptr) + ((((((((r560_z / $(128)) % $(8)) * $(4096)) + ((((r560_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r560_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r560_z / $(128)) % $(8)) * $(32)) + (((r560_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r560_z % $(128))), dtype=tl.bfloat16)
    r560_value = r560_value.to(tl.bfloat16)
    r560_value = tl.broadcast_to(r560_value, [$(32768)])
    tl.store($(candidate_vcat_94_ptr) + r560_z, r560_value)
    r561_z = tl.arange($(256))
    r561_value = tl.load($(candidate_vlogsumexp_17_ptr) + (((r561_z % $(8)) * $(32)) + ((((r561_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r561_value = r561_value.to(tl.float32)
    r561_value = tl.broadcast_to(r561_value, [$(256)])
    tl.store($(candidate_vcat_95_ptr) + r561_z, r561_value)
    r562_z = tl.arange($(49152))
    r562_value = tl.load($(candidate_vcat_92_ptr) + ((((r562_z / $(192)) % $(32)) * $(1536)) + (((r562_z / $(6144)) % $(8)) * $(192)) + (r562_z % $(192))), dtype=tl.bfloat16)
    r562_value = r562_value.to(tl.float32)
    r562_value = tl.broadcast_to(r562_value, [$(49152)])
    tl.store($(candidate_vto_copy_158_ptr) + r562_z, r562_value)
    r563_z = tl.arange($(49152))
    r563_value = tl.load($(candidate_vcat_93_ptr) + ((((r563_z / $(192)) % $(32)) * $(1536)) + (((r563_z / $(6144)) % $(8)) * $(192)) + (r563_z % $(192))), dtype=tl.bfloat16)
    r563_value = r563_value.to(tl.float32)
    r563_value = tl.broadcast_to(r563_value, [$(49152)])
    tl.store($(candidate_vto_copy_159_ptr) + r563_z, r563_value)
    r564_z = tl.arange($(32768))
    r564_value = tl.load($(candidate_vbmm_132_ptr) + ((((r564_z / $(128)) % $(32)) * $(1024)) + (((r564_z / $(4096)) % $(8)) * $(128)) + (r564_z % $(128))), dtype=tl.bfloat16)
    r564_value = r564_value.to(tl.float32)
    r564_value = tl.broadcast_to(r564_value, [$(32768)])
    tl.store($(candidate_vto_copy_160_ptr) + r564_z, r564_value)
    r565_z = tl.arange($(32768))
    r565_value = tl.load($(candidate_vmm_22_ptr) + ((((r565_z / $(128)) % $(32)) * $(1024)) + (((r565_z / $(4096)) % $(8)) * $(128)) + (r565_z % $(128))), dtype=tl.bfloat16)
    r565_value = r565_value.to(tl.float32)
    r565_value = tl.broadcast_to(r565_value, [$(32768)])
    tl.store($(candidate_vto_copy_161_ptr) + r565_z, r565_value)
    r566_z = tl.arange($(32768))
    r566_value = tl.load($(candidate_vcat_94_ptr) + ((((r566_z / $(128)) % $(32)) * $(1024)) + (((r566_z / $(4096)) % $(8)) * $(128)) + (r566_z % $(128))), dtype=tl.bfloat16)
    r566_value = r566_value.to(tl.float32)
    r566_value = tl.broadcast_to(r566_value, [$(32768)])
    tl.store($(candidate_vto_copy_162_ptr) + r566_z, r566_value)
    r567_row = tl.arange($(32))
    r567_col = tl.arange($(32))
    r567_lane = tl.arange($(192))
    for r567_group in range($(8)) {
      r567_left = tl.load($(candidate_vto_copy_158_ptr) + (((((((r567_group % $(8)) * $(6144)) + (r567_row[:, None] * $(192)) + r567_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r567_group % $(8)) * $(6144)) + (r567_row[:, None] * $(192)) + r567_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r567_group % $(8)) * $(6144)) + (r567_row[:, None] * $(192)) + r567_lane[None, :]) % $(192))), dtype=tl.float32)
      r567_right = tl.load($(candidate_vto_copy_159_ptr) + (((((((r567_group % $(8)) * $(6144)) + (r567_lane[:, None] * $(32)) + r567_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r567_group % $(8)) * $(6144)) + (r567_lane[:, None] * $(32)) + r567_col[None, :]) % $(32)) * $(192)) + (((((r567_group % $(8)) * $(6144)) + (r567_lane[:, None] * $(32)) + r567_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r567_value = tl.dot(r567_left, r567_right)
      r567_value = r567_value.to(tl.float32)
      tl.store($(candidate_vbmm_135_ptr) + ((r567_group * $(1024)) + (r567_row[:, None] * $(32)) + r567_col[None, :]), r567_value)
    }
    r568_z = tl.arange($(8192))
    r568_value = (tl.load($(candidate_vbmm_135_ptr) + ((((r568_z / $(1024)) % $(8)) * $(1024)) + (((r568_z / $(32)) % $(32)) * $(32)) + (r568_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r568_value = r568_value.to(tl.float32)
    r568_value = tl.broadcast_to(r568_value, [$(8192)])
    tl.store($(candidate_vmul_79_ptr) + r568_z, r568_value)
    r569_z = tl.arange($(32))
    r569_value = (tl.cast(r569_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r569_value = tl.broadcast_to(r569_value, [$(32)])
    tl.store($(candidate_vadd_77_ptr) + r569_z, r569_value)
    r570_z = tl.arange($(32))
    r570_value = (tl.load($(candidate_vadd_77_ptr) + r570_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r570_value = tl.broadcast_to(r570_value, [$(32)])
    tl.store($(candidate_vsub_57_ptr) + r570_z, r570_value)
    r571_z = tl.arange($(1024))
    r571_value = (tl.cast((r571_z % $(32)), tl.int64) > tl.load($(candidate_vsub_57_ptr) + ((r571_z / $(32)) % $(32)), dtype=tl.int64))
    r571_value = tl.broadcast_to(r571_value, [$(1024)])
    tl.store($(candidate_vgt_26_ptr) + r571_z, r571_value)
    r572_z = tl.arange($(8192))
    r572_value = tl.where(tl.load($(candidate_vgt_26_ptr) + ((((r572_z / $(32)) % $(32)) * $(32)) + (r572_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_79_ptr) + ((((r572_z / $(1024)) % $(8)) * $(1024)) + (((r572_z / $(32)) % $(32)) * $(32)) + (r572_z % $(32))), dtype=tl.float32))
    r572_value = r572_value.to(tl.float32)
    r572_value = tl.broadcast_to(r572_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_26_ptr) + r572_z, r572_value)
    r573_z = tl.arange($(8192))
    r573_value = (tl.load($(candidate_vmasked_fill_26_ptr) + ((((r573_z / $(1024)) % $(8)) * $(1024)) + (((r573_z / $(32)) % $(32)) * $(32)) + (r573_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_95_ptr) + ((((r573_z / $(32)) % $(32)) * $(8)) + ((r573_z / $(1024)) % $(8))), dtype=tl.float32))
    r573_value = r573_value.to(tl.float32)
    r573_value = tl.broadcast_to(r573_value, [$(8192)])
    tl.store($(candidate_vsub_58_ptr) + r573_z, r573_value)
    r574_z = tl.arange($(8192))
    r574_value = tl.exp(tl.load($(candidate_vsub_58_ptr) + ((((r574_z / $(1024)) % $(8)) * $(1024)) + (((r574_z / $(32)) % $(32)) * $(32)) + (r574_z % $(32))), dtype=tl.float32))
    r574_value = r574_value.to(tl.float32)
    r574_value = tl.broadcast_to(r574_value, [$(8192)])
    tl.store($(candidate_vexp_8_ptr) + r574_z, r574_value)
    r575_z = tl.arange($(32768))
    r575_value = (tl.load($(candidate_vto_copy_161_ptr) + ((((r575_z / $(4096)) % $(8)) * $(4096)) + (((r575_z / $(128)) % $(32)) * $(128)) + (r575_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_162_ptr) + ((((r575_z / $(4096)) % $(8)) * $(4096)) + (((r575_z / $(128)) % $(32)) * $(128)) + (r575_z % $(128))), dtype=tl.float32))
    r575_value = r575_value.to(tl.float32)
    r575_value = tl.broadcast_to(r575_value, [$(32768)])
    tl.store($(candidate_vmul_80_ptr) + r575_z, r575_value)
    r576_z = tl.arange($(256))
    r576_lane = tl.arange($(128))
    r576_value = tl.sum(tl.load($(candidate_vmul_80_ptr) + ((((r576_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r576_z[:, None] % $(32)) * $(128)) + r576_lane[None, :]), dtype=tl.float32), axis=1)
    r576_value = r576_value.to(tl.float32)
    r576_value = tl.broadcast_to(r576_value, [$(256)])
    tl.store($(candidate_vsum_17_ptr) + r576_z, r576_value)
    r577_row = tl.arange($(32))
    r577_col = tl.arange($(32))
    r577_lane = tl.arange($(128))
    for r577_group in range($(8)) {
      r577_left = tl.load($(candidate_vto_copy_161_ptr) + (((((((r577_group % $(8)) * $(4096)) + (r577_row[:, None] * $(128)) + r577_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r577_group % $(8)) * $(4096)) + (r577_row[:, None] * $(128)) + r577_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r577_group % $(8)) * $(4096)) + (r577_row[:, None] * $(128)) + r577_lane[None, :]) % $(128))), dtype=tl.float32)
      r577_right = tl.load($(candidate_vto_copy_160_ptr) + (((((((r577_group % $(8)) * $(4096)) + (r577_lane[:, None] * $(32)) + r577_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r577_group % $(8)) * $(4096)) + (r577_lane[:, None] * $(32)) + r577_col[None, :]) % $(32)) * $(128)) + (((((r577_group % $(8)) * $(4096)) + (r577_lane[:, None] * $(32)) + r577_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r577_value = tl.dot(r577_left, r577_right)
      r577_value = r577_value.to(tl.float32)
      tl.store($(candidate_vbmm_136_ptr) + ((r577_group * $(1024)) + (r577_row[:, None] * $(32)) + r577_col[None, :]), r577_value)
    }
    r578_z = tl.arange($(8192))
    r578_value = (tl.load($(candidate_vbmm_136_ptr) + ((((r578_z / $(1024)) % $(8)) * $(1024)) + (((r578_z / $(32)) % $(32)) * $(32)) + (r578_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_17_ptr) + ((((r578_z / $(1024)) % $(8)) * $(32)) + ((r578_z / $(32)) % $(32))), dtype=tl.float32))
    r578_value = r578_value.to(tl.float32)
    r578_value = tl.broadcast_to(r578_value, [$(8192)])
    tl.store($(candidate_vsub_59_ptr) + r578_z, r578_value)
    r579_z = tl.arange($(8192))
    r579_value = (tl.load($(candidate_vexp_8_ptr) + ((((r579_z / $(1024)) % $(8)) * $(1024)) + (((r579_z / $(32)) % $(32)) * $(32)) + (r579_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_59_ptr) + ((((r579_z / $(1024)) % $(8)) * $(1024)) + (((r579_z / $(32)) % $(32)) * $(32)) + (r579_z % $(32))), dtype=tl.float32))
    r579_value = r579_value.to(tl.float32)
    r579_value = tl.broadcast_to(r579_value, [$(8192)])
    tl.store($(candidate_vmul_81_ptr) + r579_z, r579_value)
    r580_row = tl.arange($(32))
    r580_col = tl.arange($(192))
    r580_lane = tl.arange($(32))
    for r580_group in range($(8)) {
      r580_left = tl.load($(candidate_vmul_81_ptr) + (((((((r580_group % $(8)) * $(1024)) + (r580_row[:, None] * $(32)) + r580_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r580_group % $(8)) * $(1024)) + (r580_row[:, None] * $(32)) + r580_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r580_group % $(8)) * $(1024)) + (r580_row[:, None] * $(32)) + r580_lane[None, :]) % $(32))), dtype=tl.float32)
      r580_right = tl.load($(candidate_vto_copy_159_ptr) + (((((((r580_group % $(8)) * $(6144)) + (r580_lane[:, None] * $(192)) + r580_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r580_group % $(8)) * $(6144)) + (r580_lane[:, None] * $(192)) + r580_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r580_group % $(8)) * $(6144)) + (r580_lane[:, None] * $(192)) + r580_col[None, :]) % $(192))), dtype=tl.float32)
      r580_value = tl.dot(r580_left, r580_right)
      r580_value = r580_value.to(tl.float32)
      tl.store($(candidate_vbmm_137_ptr) + ((r580_group * $(6144)) + (r580_row[:, None] * $(192)) + r580_col[None, :]), r580_value)
    }
    r581_z = tl.arange($(49152))
    r581_value = (tl.load($(candidate_vbmm_137_ptr) + ((((r581_z / $(6144)) % $(8)) * $(6144)) + (((r581_z / $(192)) % $(32)) * $(192)) + (r581_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r581_value = r581_value.to(tl.float32)
    r581_value = tl.broadcast_to(r581_value, [$(49152)])
    tl.store($(candidate_vmul_82_ptr) + r581_z, r581_value)
    r582_z = tl.arange($(49152))
    r582_value = tl.load($(candidate_vmul_82_ptr) + ((((r582_z / $(192)) % $(8)) * $(6144)) + (((r582_z / $(1536)) % $(32)) * $(192)) + (r582_z % $(192))), dtype=tl.float32)
    r582_value = r582_value.to(tl.bfloat16)
    r582_value = tl.broadcast_to(r582_value, [$(49152)])
    tl.store($(candidate_vto_copy_163_ptr) + r582_z, r582_value)
    r583_row = tl.arange($(32))
    r583_col = tl.arange($(192))
    r583_lane = tl.arange($(32))
    for r583_group in range($(8)) {
      r583_left = tl.load($(candidate_vmul_81_ptr) + (((((((r583_group % $(8)) * $(1024)) + (r583_row[:, None] * $(32)) + r583_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r583_group % $(8)) * $(1024)) + (r583_row[:, None] * $(32)) + r583_lane[None, :]) % $(32)) * $(32)) + (((((r583_group % $(8)) * $(1024)) + (r583_row[:, None] * $(32)) + r583_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r583_right = tl.load($(candidate_vto_copy_158_ptr) + (((((((r583_group % $(8)) * $(6144)) + (r583_lane[:, None] * $(192)) + r583_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r583_group % $(8)) * $(6144)) + (r583_lane[:, None] * $(192)) + r583_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r583_group % $(8)) * $(6144)) + (r583_lane[:, None] * $(192)) + r583_col[None, :]) % $(192))), dtype=tl.float32)
      r583_value = tl.dot(r583_left, r583_right)
      r583_value = r583_value.to(tl.float32)
      tl.store($(candidate_vbmm_138_ptr) + ((r583_group * $(6144)) + (r583_row[:, None] * $(192)) + r583_col[None, :]), r583_value)
    }
    r584_z = tl.arange($(49152))
    r584_value = (tl.load($(candidate_vbmm_138_ptr) + ((((r584_z / $(6144)) % $(8)) * $(6144)) + (((r584_z / $(192)) % $(32)) * $(192)) + (r584_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r584_value = r584_value.to(tl.float32)
    r584_value = tl.broadcast_to(r584_value, [$(49152)])
    tl.store($(candidate_vmul_83_ptr) + r584_z, r584_value)
    r585_z = tl.arange($(49152))
    r585_value = tl.load($(candidate_vmul_83_ptr) + ((((r585_z / $(192)) % $(8)) * $(6144)) + (((r585_z / $(1536)) % $(32)) * $(192)) + (r585_z % $(192))), dtype=tl.float32)
    r585_value = r585_value.to(tl.bfloat16)
    r585_value = tl.broadcast_to(r585_value, [$(49152)])
    tl.store($(candidate_vto_copy_164_ptr) + r585_z, r585_value)
    r586_row = tl.arange($(32))
    r586_col = tl.arange($(128))
    r586_lane = tl.arange($(32))
    for r586_group in range($(8)) {
      r586_left = tl.load($(candidate_vexp_8_ptr) + (((((((r586_group % $(8)) * $(1024)) + (r586_row[:, None] * $(32)) + r586_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r586_group % $(8)) * $(1024)) + (r586_row[:, None] * $(32)) + r586_lane[None, :]) % $(32)) * $(32)) + (((((r586_group % $(8)) * $(1024)) + (r586_row[:, None] * $(32)) + r586_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r586_right = tl.load($(candidate_vto_copy_161_ptr) + (((((((r586_group % $(8)) * $(4096)) + (r586_lane[:, None] * $(128)) + r586_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r586_group % $(8)) * $(4096)) + (r586_lane[:, None] * $(128)) + r586_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r586_group % $(8)) * $(4096)) + (r586_lane[:, None] * $(128)) + r586_col[None, :]) % $(128))), dtype=tl.float32)
      r586_value = tl.dot(r586_left, r586_right)
      r586_value = r586_value.to(tl.float32)
      tl.store($(candidate_vbmm_139_ptr) + ((r586_group * $(4096)) + (r586_row[:, None] * $(128)) + r586_col[None, :]), r586_value)
    }
    r587_z = tl.arange($(32768))
    r587_value = tl.load($(candidate_vbmm_139_ptr) + ((((r587_z / $(128)) % $(8)) * $(4096)) + (((r587_z / $(1024)) % $(32)) * $(128)) + (r587_z % $(128))), dtype=tl.float32)
    r587_value = r587_value.to(tl.bfloat16)
    r587_value = tl.broadcast_to(r587_value, [$(32768)])
    tl.store($(candidate_vto_copy_165_ptr) + r587_z, r587_value)
    r588_z = tl.arange($(147456))
    r588_value = tl.where(((r588_z / $(1536)) % $(96)) < $(32), tl.load($(candidate_vto_copy_164_ptr) + ((((((r588_z / $(1536)) % $(96)) - $(0)) % $(32)) * $(1536)) + (((r588_z / $(192)) % $(8)) * $(192)) + (r588_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r588_value = r588_value.to(tl.bfloat16)
    r588_value = tl.broadcast_to(r588_value, [$(147456)])
    tl.store($(candidate_vcat_96_ptr) + r588_z, r588_value)
    r589_z = tl.arange($(98304))
    r589_value = tl.where(((r589_z / $(1024)) % $(96)) < $(32), tl.load($(candidate_vto_copy_165_ptr) + ((((((r589_z / $(1024)) % $(96)) - $(0)) % $(32)) * $(1024)) + (((r589_z / $(128)) % $(8)) * $(128)) + (r589_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r589_value = r589_value.to(tl.bfloat16)
    r589_value = tl.broadcast_to(r589_value, [$(98304)])
    tl.store($(candidate_vcat_97_ptr) + r589_z, r589_value)
    r590_z = tl.arange($(49152))
    r590_value = tl.load($(candidate_vto_copy_163_ptr) + ((((((r590_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(1536)) + (((r590_z / $(192)) % $(8)) * $(192)) + (r590_z % $(192))), dtype=tl.bfloat16)
    r590_value = r590_value.to(tl.bfloat16)
    r590_value = tl.broadcast_to(r590_value, [$(49152)])
    tl.store($(candidate_vcat_98_ptr) + r590_z, r590_value)
    r591_z = tl.arange($(147456))
    r591_value = tl.load($(candidate_vcat_96_ptr) + ((((((r591_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r591_z / $(192)) % $(8)) * $(192)) + (r591_z % $(192))), dtype=tl.bfloat16)
    r591_value = r591_value.to(tl.bfloat16)
    r591_value = tl.broadcast_to(r591_value, [$(147456)])
    tl.store($(candidate_vcat_99_ptr) + r591_z, r591_value)
    r592_z = tl.arange($(98304))
    r592_value = tl.load($(candidate_vcat_97_ptr) + ((((((r592_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r592_z / $(128)) % $(8)) * $(128)) + (r592_z % $(128))), dtype=tl.bfloat16)
    r592_value = r592_value.to(tl.bfloat16)
    r592_value = tl.broadcast_to(r592_value, [$(98304)])
    tl.store($(candidate_vcat_100_ptr) + r592_z, r592_value)
    r593_z = tl.arange($(6144))
    r593_lane = tl.arange($(8))
    r593_value = tl.sum(tl.load($(candidate_vcat_99_ptr) + ((((r593_z[:, None] / $(64)) % $(96)) * $(1536)) + (r593_lane[None, :] * $(192)) + (r593_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r593_value = r593_value.to(tl.bfloat16)
    r593_value = tl.broadcast_to(r593_value, [$(6144)])
    tl.store($(candidate_vsum_18_ptr) + r593_z, r593_value)
    r594_row = tl.arange($(96))
    r594_col = tl.arange($(512))
    r594_lane = tl.arange($(1024))
    for r594_group in range($(1)) {
      r594_left = tl.load($(candidate_vcat_99_ptr) + ((((((r594_row[:, None] * $(1024)) + r594_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + ((((r594_row[:, None] * $(1024)) + r594_lane[None, :]) % $(8)) * $(192)) + ((((r594_row[:, None] * $(1024)) + r594_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r594_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r594_lane[:, None] * $(512)) + r594_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r594_lane[:, None] * $(512)) + r594_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r594_lane[:, None] * $(512)) + r594_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r594_value = tl.dot(r594_left, r594_right)
      r594_value = r594_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_140_ptr) + ((r594_group * $(49152)) + (r594_row[:, None] * $(512)) + r594_col[None, :]), r594_value)
    }
    r595_row = tl.arange($(96))
    r595_col = tl.arange($(512))
    r595_lane = tl.arange($(1024))
    for r595_group in range($(1)) {
      r595_left = tl.load($(candidate_vcat_100_ptr) + ((((((r595_row[:, None] * $(1024)) + r595_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + ((((r595_row[:, None] * $(1024)) + r595_lane[None, :]) % $(8)) * $(128)) + ((((r595_row[:, None] * $(1024)) + r595_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r595_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r595_lane[:, None] * $(512)) + r595_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r595_lane[:, None] * $(512)) + r595_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r595_lane[:, None] * $(512)) + r595_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r595_value = tl.dot(r595_left, r595_right)
      r595_value = r595_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_141_ptr) + ((r595_group * $(49152)) + (r595_row[:, None] * $(512)) + r595_col[None, :]), r595_value)
    }
    r596_z = tl.arange($(49152))
    r596_value = (tl.load($(candidate_vbmm_140_ptr) + ((((r596_z / $(512)) % $(96)) * $(512)) + (r596_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_141_ptr) + ((((r596_z / $(512)) % $(96)) * $(512)) + (r596_z % $(512))), dtype=tl.bfloat16))
    r596_value = r596_value.to(tl.bfloat16)
    r596_value = tl.broadcast_to(r596_value, [$(49152)])
    tl.store($(candidate_vadd_78_ptr) + r596_z, r596_value)
    r597_z = tl.arange($(147456))
    r597_value = tl.where(((((r597_z / $(512)) % $(288)) >= $(0)) & (((r597_z / $(512)) % $(288)) < $(96)) & ((((r597_z / $(512)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_78_ptr) + ((((((r597_z / $(512)) % $(288)) - $(0)) % $(96)) * $(512)) + (r597_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r597_value = r597_value.to(tl.bfloat16)
    r597_value = tl.broadcast_to(r597_value, [$(147456)])
    tl.store($(candidate_vslice_backward_32_ptr) + r597_z, r597_value)
    r598_z = tl.arange($(147456))
    r598_value = (tl.load($(candidate_vadd_72_ptr) + ((((r598_z / $(512)) % $(288)) * $(512)) + (r598_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_32_ptr) + ((((r598_z / $(512)) % $(288)) * $(512)) + (r598_z % $(512))), dtype=tl.bfloat16))
    r598_value = r598_value.to(tl.bfloat16)
    r598_value = tl.broadcast_to(r598_value, [$(147456)])
    tl.store($(candidate_vadd_79_ptr) + r598_z, r598_value)
    r599_z = tl.arange($(18432))
    r599_value = tl.where(((((r599_z / $(64)) % $(288)) >= $(0)) & (((r599_z / $(64)) % $(288)) < $(96)) & ((((r599_z / $(64)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_18_ptr) + ((((((r599_z / $(64)) % $(288)) - $(0)) % $(96)) * $(64)) + (r599_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r599_value = r599_value.to(tl.bfloat16)
    r599_value = tl.broadcast_to(r599_value, [$(18432)])
    tl.store($(candidate_vslice_backward_33_ptr) + r599_z, r599_value)
    r600_z = tl.arange($(18432))
    r600_value = (tl.load($(candidate_vadd_73_ptr) + ((((r600_z / $(64)) % $(288)) * $(64)) + (r600_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_33_ptr) + ((((r600_z / $(64)) % $(288)) * $(64)) + (r600_z % $(64))), dtype=tl.bfloat16))
    r600_value = r600_value.to(tl.bfloat16)
    r600_value = tl.broadcast_to(r600_value, [$(18432)])
    tl.store($(candidate_vadd_80_ptr) + r600_z, r600_value)
    r601_z = tl.arange($(294912))
    r601_value = tl.where(((((r601_z / $(1024)) % $(288)) >= $(0)) & (((r601_z / $(1024)) % $(288)) < $(32)) & ((((r601_z / $(1024)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vcat_98_ptr) + ((((((r601_z / $(1024)) % $(288)) - $(0)) % $(32)) * $(1536)) + (((r601_z / $(128)) % $(8)) * $(192)) + (r601_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r601_value = r601_value.to(tl.bfloat16)
    r601_value = tl.broadcast_to(r601_value, [$(294912)])
    tl.store($(candidate_vslice_backward_34_ptr) + r601_z, r601_value)
    r602_z = tl.arange($(294912))
    r602_value = (tl.load($(candidate_vadd_74_ptr) + ((((r602_z / $(1024)) % $(288)) * $(1024)) + (((r602_z / $(128)) % $(8)) * $(128)) + (r602_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_34_ptr) + ((((r602_z / $(1024)) % $(288)) * $(1024)) + (((r602_z / $(128)) % $(8)) * $(128)) + (r602_z % $(128))), dtype=tl.bfloat16))
    r602_value = r602_value.to(tl.bfloat16)
    r602_value = tl.broadcast_to(r602_value, [$(294912)])
    tl.store($(candidate_vadd_81_ptr) + r602_z, r602_value)
    r603_z = tl.arange($(147456))
    r603_value = tl.where(((((r603_z / $(512)) % $(288)) >= $(0)) & (((r603_z / $(512)) % $(288)) < $(32)) & ((((r603_z / $(512)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vcat_98_ptr) + ((((((r603_z / $(512)) % $(288)) - $(0)) % $(32)) * $(1536)) + (((r603_z / $(64)) % $(8)) * $(192)) + (r603_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r603_value = r603_value.to(tl.bfloat16)
    r603_value = tl.broadcast_to(r603_value, [$(147456)])
    tl.store($(candidate_vslice_backward_35_ptr) + r603_z, r603_value)
    r604_z = tl.arange($(147456))
    r604_value = (tl.load($(candidate_vadd_75_ptr) + ((((r604_z / $(512)) % $(288)) * $(512)) + (((r604_z / $(64)) % $(8)) * $(64)) + (r604_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_35_ptr) + ((((r604_z / $(512)) % $(288)) * $(512)) + (((r604_z / $(64)) % $(8)) * $(64)) + (r604_z % $(64))), dtype=tl.bfloat16))
    r604_value = r604_value.to(tl.bfloat16)
    r604_value = tl.broadcast_to(r604_value, [$(147456)])
    tl.store($(candidate_vadd_82_ptr) + r604_z, r604_value)
    r605_z = tl.arange($(165888))
    r605_value = tl.where((r605_z % $(576)) < $(512), tl.load($(candidate_vadd_79_ptr) + ((((r605_z / $(576)) % $(288)) * $(512)) + (((r605_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_80_ptr) + ((((r605_z / $(576)) % $(288)) * $(64)) + (((r605_z % $(576)) - $(512)) % $(64))), dtype=tl.bfloat16))
    r605_value = r605_value.to(tl.bfloat16)
    r605_value = tl.broadcast_to(r605_value, [$(165888)])
    tl.store($(candidate_vcat_102_ptr) + r605_z, r605_value)
    r606_z = tl.arange($(9216))
    r606_value = (tl.load($(candidate_vcat_102_ptr) + ((((r606_z / $(32)) % $(288)) * $(576)) + (r606_z % $(32)) + $(544)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r606_z / $(32)) % $(288)) * $(64)) + (r606_z % $(32))), dtype=tl.bfloat16))
    r606_value = r606_value.to(tl.bfloat16)
    r606_value = tl.broadcast_to(r606_value, [$(9216)])
    tl.store($(candidate_vmul_84_ptr) + r606_z, r606_value)
    r607_z = tl.arange($(9216))
    r607_value = (tl.load($(candidate_vcat_102_ptr) + ((((r607_z / $(32)) % $(288)) * $(576)) + (r607_z % $(32)) + $(544)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r607_z / $(32)) % $(288)) * $(64)) + (r607_z % $(32))), dtype=tl.bfloat16))
    r607_value = r607_value.to(tl.bfloat16)
    r607_value = tl.broadcast_to(r607_value, [$(9216)])
    tl.store($(candidate_vmul_85_ptr) + r607_z, r607_value)
    r608_z = tl.arange($(9216))
    r608_value = -tl.load($(candidate_vcat_102_ptr) + ((((r608_z / $(32)) % $(288)) * $(576)) + (r608_z % $(32)) + $(512)), dtype=tl.bfloat16)
    r608_value = r608_value.to(tl.bfloat16)
    r608_value = tl.broadcast_to(r608_value, [$(9216)])
    tl.store($(candidate_vneg_ptr) + r608_z, r608_value)
    r609_z = tl.arange($(9216))
    r609_value = (tl.load($(candidate_vneg_ptr) + ((((r609_z / $(32)) % $(288)) * $(32)) + (r609_z % $(32))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r609_z / $(32)) % $(288)) * $(64)) + (r609_z % $(32))), dtype=tl.bfloat16))
    r609_value = r609_value.to(tl.bfloat16)
    r609_value = tl.broadcast_to(r609_value, [$(9216)])
    tl.store($(candidate_vmul_86_ptr) + r609_z, r609_value)
    r610_z = tl.arange($(9216))
    r610_value = (tl.load($(candidate_vmul_85_ptr) + ((((r610_z / $(32)) % $(288)) * $(32)) + (r610_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_86_ptr) + ((((r610_z / $(32)) % $(288)) * $(32)) + (r610_z % $(32))), dtype=tl.bfloat16))
    r610_value = r610_value.to(tl.bfloat16)
    r610_value = tl.broadcast_to(r610_value, [$(9216)])
    tl.store($(candidate_vadd_83_ptr) + r610_z, r610_value)
    r611_z = tl.arange($(9216))
    r611_value = (tl.load($(candidate_vcat_102_ptr) + ((((r611_z / $(32)) % $(288)) * $(576)) + (r611_z % $(32)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r611_z / $(32)) % $(288)) * $(64)) + (r611_z % $(32))), dtype=tl.bfloat16))
    r611_value = r611_value.to(tl.bfloat16)
    r611_value = tl.broadcast_to(r611_value, [$(9216)])
    tl.store($(candidate_vmul_87_ptr) + r611_z, r611_value)
    r612_z = tl.arange($(9216))
    r612_value = (tl.load($(candidate_vmul_84_ptr) + ((((r612_z / $(32)) % $(288)) * $(32)) + (r612_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_87_ptr) + ((((r612_z / $(32)) % $(288)) * $(32)) + (r612_z % $(32))), dtype=tl.bfloat16))
    r612_value = r612_value.to(tl.bfloat16)
    r612_value = tl.broadcast_to(r612_value, [$(9216)])
    tl.store($(candidate_vadd_84_ptr) + r612_z, r612_value)
    r613_z = tl.arange($(73728))
    r613_value = (tl.load($(candidate_vadd_82_ptr) + ((((r613_z / $(32)) % $(288)) * $(512)) + (((r613_z / $(9216)) % $(8)) * $(64)) + (r613_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r613_z / $(32)) % $(288)) * $(64)) + (r613_z % $(32))), dtype=tl.bfloat16))
    r613_value = r613_value.to(tl.bfloat16)
    r613_value = tl.broadcast_to(r613_value, [$(73728)])
    tl.store($(candidate_vmul_88_ptr) + r613_z, r613_value)
    r614_z = tl.arange($(73728))
    r614_value = (tl.load($(candidate_vadd_82_ptr) + ((((r614_z / $(32)) % $(288)) * $(512)) + (((r614_z / $(9216)) % $(8)) * $(64)) + (r614_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r614_z / $(32)) % $(288)) * $(64)) + (r614_z % $(32))), dtype=tl.bfloat16))
    r614_value = r614_value.to(tl.bfloat16)
    r614_value = tl.broadcast_to(r614_value, [$(73728)])
    tl.store($(candidate_vmul_89_ptr) + r614_z, r614_value)
    r615_z = tl.arange($(73728))
    r615_value = -tl.load($(candidate_vadd_82_ptr) + ((((r615_z / $(32)) % $(288)) * $(512)) + (((r615_z / $(9216)) % $(8)) * $(64)) + (r615_z % $(32))), dtype=tl.bfloat16)
    r615_value = r615_value.to(tl.bfloat16)
    r615_value = tl.broadcast_to(r615_value, [$(73728)])
    tl.store($(candidate_vneg_1_ptr) + r615_z, r615_value)
    r616_z = tl.arange($(73728))
    r616_value = (tl.load($(candidate_vneg_1_ptr) + ((((r616_z / $(9216)) % $(8)) * $(9216)) + (((r616_z / $(32)) % $(288)) * $(32)) + (r616_z % $(32))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r616_z / $(32)) % $(288)) * $(64)) + (r616_z % $(32))), dtype=tl.bfloat16))
    r616_value = r616_value.to(tl.bfloat16)
    r616_value = tl.broadcast_to(r616_value, [$(73728)])
    tl.store($(candidate_vmul_90_ptr) + r616_z, r616_value)
    r617_z = tl.arange($(73728))
    r617_value = (tl.load($(candidate_vmul_89_ptr) + ((((r617_z / $(9216)) % $(8)) * $(9216)) + (((r617_z / $(32)) % $(288)) * $(32)) + (r617_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_90_ptr) + ((((r617_z / $(9216)) % $(8)) * $(9216)) + (((r617_z / $(32)) % $(288)) * $(32)) + (r617_z % $(32))), dtype=tl.bfloat16))
    r617_value = r617_value.to(tl.bfloat16)
    r617_value = tl.broadcast_to(r617_value, [$(73728)])
    tl.store($(candidate_vadd_85_ptr) + r617_z, r617_value)
    r618_z = tl.arange($(73728))
    r618_value = (tl.load($(candidate_vadd_82_ptr) + ((((r618_z / $(32)) % $(288)) * $(512)) + (((r618_z / $(9216)) % $(8)) * $(64)) + (r618_z % $(32))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r618_z / $(32)) % $(288)) * $(64)) + (r618_z % $(32))), dtype=tl.bfloat16))
    r618_value = r618_value.to(tl.bfloat16)
    r618_value = tl.broadcast_to(r618_value, [$(73728)])
    tl.store($(candidate_vmul_91_ptr) + r618_z, r618_value)
    r619_z = tl.arange($(73728))
    r619_value = (tl.load($(candidate_vmul_88_ptr) + ((((r619_z / $(9216)) % $(8)) * $(9216)) + (((r619_z / $(32)) % $(288)) * $(32)) + (r619_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_91_ptr) + ((((r619_z / $(9216)) % $(8)) * $(9216)) + (((r619_z / $(32)) % $(288)) * $(32)) + (r619_z % $(32))), dtype=tl.bfloat16))
    r619_value = r619_value.to(tl.bfloat16)
    r619_value = tl.broadcast_to(r619_value, [$(73728)])
    tl.store($(candidate_vadd_86_ptr) + r619_z, r619_value)
    r620_z = tl.arange($(18432))
    r620_value = tl.where((((r620_z % $(64)) >= $(1)) & ((r620_z % $(64)) < $(64)) & (((r620_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_83_ptr) + ((((r620_z / $(64)) % $(288)) * $(32)) + ((((r620_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r620_value = r620_value.to(tl.bfloat16)
    r620_value = tl.broadcast_to(r620_value, [$(18432)])
    tl.store($(candidate_vslice_backward_36_ptr) + r620_z, r620_value)
    r621_z = tl.arange($(18432))
    r621_value = tl.where((((r621_z % $(64)) >= $(0)) & ((r621_z % $(64)) < $(64)) & (((r621_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_84_ptr) + ((((r621_z / $(64)) % $(288)) * $(32)) + ((((r621_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r621_value = r621_value.to(tl.bfloat16)
    r621_value = tl.broadcast_to(r621_value, [$(18432)])
    tl.store($(candidate_vslice_backward_37_ptr) + r621_z, r621_value)
    r622_z = tl.arange($(18432))
    r622_value = (tl.load($(candidate_vslice_backward_36_ptr) + ((((r622_z / $(64)) % $(288)) * $(64)) + (r622_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_37_ptr) + ((((r622_z / $(64)) % $(288)) * $(64)) + (r622_z % $(64))), dtype=tl.bfloat16))
    r622_value = r622_value.to(tl.bfloat16)
    r622_value = tl.broadcast_to(r622_value, [$(18432)])
    tl.store($(candidate_vadd_87_ptr) + r622_z, r622_value)
    r623_z = tl.arange($(147456))
    r623_value = tl.where((((r623_z % $(64)) >= $(1)) & ((r623_z % $(64)) < $(64)) & (((r623_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_85_ptr) + ((((r623_z / $(18432)) % $(8)) * $(9216)) + (((r623_z / $(64)) % $(288)) * $(32)) + ((((r623_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r623_value = r623_value.to(tl.bfloat16)
    r623_value = tl.broadcast_to(r623_value, [$(147456)])
    tl.store($(candidate_vslice_backward_38_ptr) + r623_z, r623_value)
    r624_z = tl.arange($(147456))
    r624_value = tl.where((((r624_z % $(64)) >= $(0)) & ((r624_z % $(64)) < $(64)) & (((r624_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_86_ptr) + ((((r624_z / $(18432)) % $(8)) * $(9216)) + (((r624_z / $(64)) % $(288)) * $(32)) + ((((r624_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r624_value = r624_value.to(tl.bfloat16)
    r624_value = tl.broadcast_to(r624_value, [$(147456)])
    tl.store($(candidate_vslice_backward_39_ptr) + r624_z, r624_value)
    r625_z = tl.arange($(147456))
    r625_value = (tl.load($(candidate_vslice_backward_38_ptr) + ((((r625_z / $(18432)) % $(8)) * $(18432)) + (((r625_z / $(64)) % $(288)) * $(64)) + (r625_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_39_ptr) + ((((r625_z / $(18432)) % $(8)) * $(18432)) + (((r625_z / $(64)) % $(288)) * $(64)) + (r625_z % $(64))), dtype=tl.bfloat16))
    r625_value = r625_value.to(tl.bfloat16)
    r625_value = tl.broadcast_to(r625_value, [$(147456)])
    tl.store($(candidate_vadd_88_ptr) + r625_z, r625_value)
    r626_z = tl.arange($(442368))
    r626_value = tl.where((r626_z % $(192)) < $(128), tl.load($(candidate_vadd_81_ptr) + ((((((((r626_z / $(192)) % $(288)) * $(1024)) + (((r626_z / $(55296)) % $(8)) * $(128)) + ((r626_z % $(192)) - $(0))) / $(1024)) % $(288)) * $(1024)) + (((((((r626_z / $(192)) % $(288)) * $(1024)) + (((r626_z / $(55296)) % $(8)) * $(128)) + ((r626_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r626_z / $(192)) % $(288)) * $(1024)) + (((r626_z / $(55296)) % $(8)) * $(128)) + ((r626_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_88_ptr) + ((((r626_z / $(55296)) % $(8)) * $(18432)) + (((r626_z / $(192)) % $(288)) * $(64)) + (((r626_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r626_value = r626_value.to(tl.bfloat16)
    r626_value = tl.broadcast_to(r626_value, [$(442368)])
    tl.store($(candidate_vcat_103_ptr) + r626_z, r626_value)
    r627_z = tl.arange($(147456))
    r627_value = tl.load($(candidate_vcat_102_ptr) + ((((r627_z / $(512)) % $(288)) * $(576)) + (r627_z % $(512))), dtype=tl.bfloat16)
    r627_value = r627_value.to(tl.float32)
    r627_value = tl.broadcast_to(r627_value, [$(147456)])
    tl.store($(candidate_vto_copy_166_ptr) + r627_z, r627_value)
    r628_z = tl.arange($(147456))
    r628_value = (tl.load($(candidate_vto_copy_166_ptr) + ((((r628_z / $(512)) % $(288)) * $(512)) + (r628_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r628_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r628_value = r628_value.to(tl.float32)
    r628_value = tl.broadcast_to(r628_value, [$(147456)])
    tl.store($(candidate_vmul_92_ptr) + r628_z, r628_value)
    r629_z = tl.arange($(147456))
    r629_value = (tl.load($(candidate_vmul_92_ptr) + ((((r629_z / $(512)) % $(288)) * $(512)) + (r629_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r629_z / $(512)) % $(288)) * $(512)) + (r629_z % $(512))), dtype=tl.float32))
    r629_value = r629_value.to(tl.float32)
    r629_value = tl.broadcast_to(r629_value, [$(147456)])
    tl.store($(candidate_vmul_93_ptr) + r629_z, r629_value)
    r630_z = tl.arange($(147456))
    r630_value = (tl.load($(candidate_vmul_92_ptr) + ((((r630_z / $(512)) % $(288)) * $(512)) + (r630_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r630_z / $(512)) % $(288)), dtype=tl.float32))
    r630_value = r630_value.to(tl.float32)
    r630_value = tl.broadcast_to(r630_value, [$(147456)])
    tl.store($(candidate_vmul_94_ptr) + r630_z, r630_value)
    r631_z = tl.arange($(288))
    r631_lane = tl.arange($(512))
    r631_value = tl.sum(tl.load($(candidate_vmul_93_ptr) + ((r631_z[:, None] * $(512)) + r631_lane[None, :]), dtype=tl.float32), axis=1)
    r631_value = r631_value.to(tl.float32)
    r631_value = tl.broadcast_to(r631_value, [$(288)])
    tl.store($(candidate_vsum_19_ptr) + r631_z, r631_value)
    r632_z = tl.arange($(288))
    r632_value = ((tl.load($(candidate_vrsqrt_1_ptr) + r632_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r632_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r632_z, dtype=tl.float32)))
    r632_value = r632_value.to(tl.float32)
    r632_value = tl.broadcast_to(r632_value, [$(288)])
    tl.store($(candidate_vpow_4_ptr) + r632_z, r632_value)
    r633_z = tl.arange($(288))
    r633_value = (tl.load($(candidate_vsum_19_ptr) + r633_z, dtype=tl.float32) * -0.5)
    r633_value = r633_value.to(tl.float32)
    r633_value = tl.broadcast_to(r633_value, [$(288)])
    tl.store($(candidate_vmul_95_ptr) + r633_z, r633_value)
    r634_z = tl.arange($(288))
    r634_value = (tl.load($(candidate_vmul_95_ptr) + r634_z, dtype=tl.float32) * tl.load($(candidate_vpow_4_ptr) + r634_z, dtype=tl.float32))
    r634_value = r634_value.to(tl.float32)
    r634_value = tl.broadcast_to(r634_value, [$(288)])
    tl.store($(candidate_vmul_96_ptr) + r634_z, r634_value)
    r635_z = tl.arange($(147456))
    r635_value = (tl.load($(candidate_vmul_96_ptr) + ((r635_z / $(512)) % $(288)), dtype=tl.float32) / 512.0)
    r635_value = r635_value.to(tl.float32)
    r635_value = tl.broadcast_to(r635_value, [$(147456)])
    tl.store($(candidate_vdiv_1_ptr) + r635_z, r635_value)
    r636_z = tl.arange($(147456))
    r636_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r636_z / $(512)) % $(288)) * $(512)) + (r636_z % $(512))), dtype=tl.float32)))
    r636_value = r636_value.to(tl.float32)
    r636_value = tl.broadcast_to(r636_value, [$(147456)])
    tl.store($(candidate_vpow_5_ptr) + r636_z, r636_value)
    r637_z = tl.arange($(147456))
    r637_value = (tl.load($(candidate_vpow_5_ptr) + ((((r637_z / $(512)) % $(288)) * $(512)) + (r637_z % $(512))), dtype=tl.float32) * 2.0)
    r637_value = r637_value.to(tl.float32)
    r637_value = tl.broadcast_to(r637_value, [$(147456)])
    tl.store($(candidate_vmul_97_ptr) + r637_z, r637_value)
    r638_z = tl.arange($(147456))
    r638_value = (tl.load($(candidate_vdiv_1_ptr) + ((((r638_z / $(512)) % $(288)) * $(512)) + (r638_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vmul_97_ptr) + ((((r638_z / $(512)) % $(288)) * $(512)) + (r638_z % $(512))), dtype=tl.float32))
    r638_value = r638_value.to(tl.float32)
    r638_value = tl.broadcast_to(r638_value, [$(147456)])
    tl.store($(candidate_vmul_98_ptr) + r638_z, r638_value)
    r639_z = tl.arange($(147456))
    r639_value = (tl.load($(candidate_vmul_94_ptr) + ((((r639_z / $(512)) % $(288)) * $(512)) + (r639_z % $(512))), dtype=tl.float32) + tl.load($(candidate_vmul_98_ptr) + ((((r639_z / $(512)) % $(288)) * $(512)) + (r639_z % $(512))), dtype=tl.float32))
    r639_value = r639_value.to(tl.float32)
    r639_value = tl.broadcast_to(r639_value, [$(147456)])
    tl.store($(candidate_vadd_89_ptr) + r639_z, r639_value)
    r640_z = tl.arange($(147456))
    r640_value = tl.load($(candidate_vadd_89_ptr) + ((((r640_z / $(512)) % $(288)) * $(512)) + (r640_z % $(512))), dtype=tl.float32)
    r640_value = r640_value.to(tl.bfloat16)
    r640_value = tl.broadcast_to(r640_value, [$(147456)])
    tl.store($(candidate_vto_copy_167_ptr) + r640_z, r640_value)
    r641_z = tl.arange($(165888))
    r641_value = tl.where((r641_z % $(576)) < $(512), tl.load($(candidate_vto_copy_167_ptr) + ((((r641_z / $(576)) % $(288)) * $(512)) + (((r641_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_87_ptr) + ((((((((r641_z / $(576)) % $(288)) * $(64)) + ((r641_z % $(576)) - $(512))) / $(64)) % $(288)) * $(64)) + (((((r641_z / $(576)) % $(288)) * $(64)) + ((r641_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r641_value = r641_value.to(tl.bfloat16)
    r641_value = tl.broadcast_to(r641_value, [$(165888)])
    tl.store($(candidate_vcat_104_ptr) + r641_z, r641_value)
    r642_row = tl.arange($(288))
    r642_col = tl.arange($(512))
    r642_lane = tl.arange($(576))
    r642_left = tl.load($(candidate_vcat_104_ptr) + ((((((r642_row[:, None] * $(576)) + r642_lane[None, :]) / $(576)) % $(288)) * $(576)) + (((r642_row[:, None] * $(576)) + r642_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r642_right = tl.load($(candidate_vto_copy_3_ptr) + (((r642_lane[:, None] % $(576)) * $(512)) + (r642_col[None, :] % $(512))), dtype=tl.bfloat16)
    r642_value = tl.dot(r642_left, r642_right)
    r642_value = r642_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_23_ptr) + ((r642_row[:, None] * $(512)) + r642_col[None, :]), r642_value)
    r643_row = tl.arange($(288))
    r643_col = tl.arange($(256))
    r643_lane = tl.arange($(1536))
    r643_left = tl.load($(candidate_vcat_103_ptr) + ((((((((((r643_row[:, None] * $(1536)) + r643_lane[None, :]) / $(1536)) % $(288)) * $(1536)) + (((r643_row[:, None] * $(1536)) + r643_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(55296)) + (((((r643_row[:, None] * $(1536)) + r643_lane[None, :]) / $(1536)) % $(288)) * $(192)) + (((((((r643_row[:, None] * $(1536)) + r643_lane[None, :]) / $(1536)) % $(288)) * $(1536)) + (((r643_row[:, None] * $(1536)) + r643_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r643_right = tl.load($(candidate_vto_copy_2_ptr) + (((r643_lane[:, None] % $(1536)) * $(256)) + (r643_col[None, :] % $(256))), dtype=tl.bfloat16)
    r643_value = tl.dot(r643_left, r643_right)
    r643_value = r643_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_24_ptr) + ((r643_row[:, None] * $(256)) + r643_col[None, :]), r643_value)
    r644_z = tl.arange($(73728))
    r644_value = tl.load($(candidate_vmm_24_ptr) + ((((r644_z / $(256)) % $(288)) * $(256)) + (r644_z % $(256))), dtype=tl.bfloat16)
    r644_value = r644_value.to(tl.float32)
    r644_value = tl.broadcast_to(r644_value, [$(73728)])
    tl.store($(candidate_vto_copy_168_ptr) + r644_z, r644_value)
    r645_z = tl.arange($(73728))
    r645_value = (tl.load($(candidate_vto_copy_168_ptr) + ((((r645_z / $(256)) % $(288)) * $(256)) + (r645_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r645_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r645_value = r645_value.to(tl.float32)
    r645_value = tl.broadcast_to(r645_value, [$(73728)])
    tl.store($(candidate_vmul_99_ptr) + r645_z, r645_value)
    r646_z = tl.arange($(73728))
    r646_value = (tl.load($(candidate_vmul_99_ptr) + ((((r646_z / $(256)) % $(288)) * $(256)) + (r646_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_18_ptr) + ((((r646_z / $(256)) % $(288)) * $(256)) + (r646_z % $(256))), dtype=tl.float32))
    r646_value = r646_value.to(tl.float32)
    r646_value = tl.broadcast_to(r646_value, [$(73728)])
    tl.store($(candidate_vmul_100_ptr) + r646_z, r646_value)
    r647_z = tl.arange($(73728))
    r647_value = (tl.load($(candidate_vmul_99_ptr) + ((((r647_z / $(256)) % $(288)) * $(256)) + (r647_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r647_z / $(256)) % $(288)), dtype=tl.float32))
    r647_value = r647_value.to(tl.float32)
    r647_value = tl.broadcast_to(r647_value, [$(73728)])
    tl.store($(candidate_vmul_101_ptr) + r647_z, r647_value)
    r648_z = tl.arange($(288))
    r648_lane = tl.arange($(256))
    r648_value = tl.sum(tl.load($(candidate_vmul_100_ptr) + ((r648_z[:, None] * $(256)) + r648_lane[None, :]), dtype=tl.float32), axis=1)
    r648_value = r648_value.to(tl.float32)
    r648_value = tl.broadcast_to(r648_value, [$(288)])
    tl.store($(candidate_vsum_20_ptr) + r648_z, r648_value)
    r649_z = tl.arange($(288))
    r649_value = ((tl.load($(candidate_vrsqrt_ptr) + r649_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r649_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r649_z, dtype=tl.float32)))
    r649_value = r649_value.to(tl.float32)
    r649_value = tl.broadcast_to(r649_value, [$(288)])
    tl.store($(candidate_vpow_6_ptr) + r649_z, r649_value)
    r650_z = tl.arange($(288))
    r650_value = (tl.load($(candidate_vsum_20_ptr) + r650_z, dtype=tl.float32) * -0.5)
    r650_value = r650_value.to(tl.float32)
    r650_value = tl.broadcast_to(r650_value, [$(288)])
    tl.store($(candidate_vmul_102_ptr) + r650_z, r650_value)
    r651_z = tl.arange($(288))
    r651_value = (tl.load($(candidate_vmul_102_ptr) + r651_z, dtype=tl.float32) * tl.load($(candidate_vpow_6_ptr) + r651_z, dtype=tl.float32))
    r651_value = r651_value.to(tl.float32)
    r651_value = tl.broadcast_to(r651_value, [$(288)])
    tl.store($(candidate_vmul_103_ptr) + r651_z, r651_value)
    r652_z = tl.arange($(73728))
    r652_value = (tl.load($(candidate_vmul_103_ptr) + ((r652_z / $(256)) % $(288)), dtype=tl.float32) / 256.0)
    r652_value = r652_value.to(tl.float32)
    r652_value = tl.broadcast_to(r652_value, [$(73728)])
    tl.store($(candidate_vdiv_2_ptr) + r652_z, r652_value)
    r653_z = tl.arange($(73728))
    r653_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r653_z / $(256)) % $(288)) * $(256)) + (r653_z % $(256))), dtype=tl.float32)))
    r653_value = r653_value.to(tl.float32)
    r653_value = tl.broadcast_to(r653_value, [$(73728)])
    tl.store($(candidate_vpow_7_ptr) + r653_z, r653_value)
    r654_z = tl.arange($(73728))
    r654_value = (tl.load($(candidate_vpow_7_ptr) + ((((r654_z / $(256)) % $(288)) * $(256)) + (r654_z % $(256))), dtype=tl.float32) * 2.0)
    r654_value = r654_value.to(tl.float32)
    r654_value = tl.broadcast_to(r654_value, [$(73728)])
    tl.store($(candidate_vmul_104_ptr) + r654_z, r654_value)
    r655_z = tl.arange($(73728))
    r655_value = (tl.load($(candidate_vdiv_2_ptr) + ((((r655_z / $(256)) % $(288)) * $(256)) + (r655_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_104_ptr) + ((((r655_z / $(256)) % $(288)) * $(256)) + (r655_z % $(256))), dtype=tl.float32))
    r655_value = r655_value.to(tl.float32)
    r655_value = tl.broadcast_to(r655_value, [$(73728)])
    tl.store($(candidate_vmul_105_ptr) + r655_z, r655_value)
    r656_z = tl.arange($(73728))
    r656_value = (tl.load($(candidate_vmul_101_ptr) + ((((r656_z / $(256)) % $(288)) * $(256)) + (r656_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_105_ptr) + ((((r656_z / $(256)) % $(288)) * $(256)) + (r656_z % $(256))), dtype=tl.float32))
    r656_value = r656_value.to(tl.float32)
    r656_value = tl.broadcast_to(r656_value, [$(73728)])
    tl.store($(candidate_vadd_90_ptr) + r656_z, r656_value)
    r657_z = tl.arange($(73728))
    r657_value = tl.load($(candidate_vadd_90_ptr) + ((((r657_z / $(256)) % $(288)) * $(256)) + (r657_z % $(256))), dtype=tl.float32)
    r657_value = r657_value.to(tl.bfloat16)
    r657_value = tl.broadcast_to(r657_value, [$(73728)])
    tl.store($(candidate_vto_copy_169_ptr) + r657_z, r657_value)
    r658_row = tl.arange($(288))
    r658_col = tl.arange($(512))
    r658_lane = tl.arange($(256))
    r658_left = tl.load($(candidate_vto_copy_169_ptr) + ((((((r658_row[:, None] * $(256)) + r658_lane[None, :]) / $(256)) % $(288)) * $(256)) + (((r658_row[:, None] * $(256)) + r658_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r658_right = tl.load($(candidate_vto_copy_ptr) + (((r658_lane[:, None] % $(256)) * $(512)) + (r658_col[None, :] % $(512))), dtype=tl.bfloat16)
    r658_value = tl.dot(r658_left, r658_right)
    r658_value = r658_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_25_ptr) + ((r658_row[:, None] * $(512)) + r658_col[None, :]), r658_value)
    r659_z = tl.arange($(147456))
    r659_value = (tl.load($(candidate_vmm_23_ptr) + ((((r659_z / $(512)) % $(288)) * $(512)) + (r659_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_25_ptr) + ((((r659_z / $(512)) % $(288)) * $(512)) + (r659_z % $(512))), dtype=tl.bfloat16))
    r659_value = r659_value.to(tl.bfloat16)
    r659_value = tl.broadcast_to(r659_value, [$(147456)])
    tl.store($(candidate_vadd_91_ptr) + r659_z, r659_value)
    z = tl.arange($(147456))
    value = tl.load($(candidate_vadd_91_ptr) + ((((z / $(512)) % $(288)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(147456)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_seqused_dx
