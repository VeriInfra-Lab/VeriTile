import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_seqused_dx
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
  let vadd_15_ptr : RegionName := "vadd_15"
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
  let candidate_vindex_put_ptr : RegionName := "candidate_vindex_put"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vbmm_36_ptr : RegionName := "candidate_vbmm_36"
  let candidate_vbmm_37_ptr : RegionName := "candidate_vbmm_37"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vsoftmax_backward_data_ptr : RegionName := "candidate_vsoftmax_backward_data"
  let candidate_vmasked_fill_9_ptr : RegionName := "candidate_vmasked_fill_9"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vbmm_38_ptr : RegionName := "candidate_vbmm_38"
  let candidate_vbmm_39_ptr : RegionName := "candidate_vbmm_39"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vfill_ptr : RegionName := "candidate_vfill"
  let candidate_vslice_scatter_ptr : RegionName := "candidate_vslice_scatter"
  let candidate_vfill_1_ptr : RegionName := "candidate_vfill_1"
  let candidate_vslice_scatter_1_ptr : RegionName := "candidate_vslice_scatter_1"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_40_ptr : RegionName := "candidate_vbmm_40"
  let candidate_vbmm_41_ptr : RegionName := "candidate_vbmm_41"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vselect_backward_ptr : RegionName := "candidate_vselect_backward"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vbmm_42_ptr : RegionName := "candidate_vbmm_42"
  let candidate_vbmm_43_ptr : RegionName := "candidate_vbmm_43"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vsoftmax_backward_data_1_ptr : RegionName := "candidate_vsoftmax_backward_data_1"
  let candidate_vmasked_fill_10_ptr : RegionName := "candidate_vmasked_fill_10"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vbmm_44_ptr : RegionName := "candidate_vbmm_44"
  let candidate_vbmm_45_ptr : RegionName := "candidate_vbmm_45"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vslice_backward_8_ptr : RegionName := "candidate_vslice_backward_8"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vslice_backward_9_ptr : RegionName := "candidate_vslice_backward_9"
  let candidate_vfill_2_ptr : RegionName := "candidate_vfill_2"
  let candidate_vslice_scatter_2_ptr : RegionName := "candidate_vslice_scatter_2"
  let candidate_vfill_3_ptr : RegionName := "candidate_vfill_3"
  let candidate_vslice_scatter_3_ptr : RegionName := "candidate_vslice_scatter_3"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vbmm_46_ptr : RegionName := "candidate_vbmm_46"
  let candidate_vbmm_47_ptr : RegionName := "candidate_vbmm_47"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vselect_backward_1_ptr : RegionName := "candidate_vselect_backward_1"
  let candidate_vslice_backward_10_ptr : RegionName := "candidate_vslice_backward_10"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vslice_backward_11_ptr : RegionName := "candidate_vslice_backward_11"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vslice_backward_12_ptr : RegionName := "candidate_vslice_backward_12"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vslice_backward_13_ptr : RegionName := "candidate_vslice_backward_13"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vbmm_48_ptr : RegionName := "candidate_vbmm_48"
  let candidate_vbmm_49_ptr : RegionName := "candidate_vbmm_49"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vsoftmax_backward_data_2_ptr : RegionName := "candidate_vsoftmax_backward_data_2"
  let candidate_vmasked_fill_11_ptr : RegionName := "candidate_vmasked_fill_11"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vbmm_50_ptr : RegionName := "candidate_vbmm_50"
  let candidate_vbmm_51_ptr : RegionName := "candidate_vbmm_51"
  let candidate_vslice_backward_14_ptr : RegionName := "candidate_vslice_backward_14"
  let candidate_vto_copy_56_ptr : RegionName := "candidate_vto_copy_56"
  let candidate_vslice_backward_15_ptr : RegionName := "candidate_vslice_backward_15"
  let candidate_vto_copy_57_ptr : RegionName := "candidate_vto_copy_57"
  let candidate_vslice_backward_16_ptr : RegionName := "candidate_vslice_backward_16"
  let candidate_vfill_4_ptr : RegionName := "candidate_vfill_4"
  let candidate_vslice_scatter_4_ptr : RegionName := "candidate_vslice_scatter_4"
  let candidate_vfill_5_ptr : RegionName := "candidate_vfill_5"
  let candidate_vslice_scatter_5_ptr : RegionName := "candidate_vslice_scatter_5"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vbmm_52_ptr : RegionName := "candidate_vbmm_52"
  let candidate_vbmm_53_ptr : RegionName := "candidate_vbmm_53"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vselect_backward_2_ptr : RegionName := "candidate_vselect_backward_2"
  let candidate_vslice_backward_17_ptr : RegionName := "candidate_vslice_backward_17"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vslice_backward_18_ptr : RegionName := "candidate_vslice_backward_18"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
  let candidate_vslice_backward_19_ptr : RegionName := "candidate_vslice_backward_19"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vslice_backward_20_ptr : RegionName := "candidate_vslice_backward_20"
  let candidate_vadd_25_ptr : RegionName := "candidate_vadd_25"
  let candidate_vbmm_54_ptr : RegionName := "candidate_vbmm_54"
  let candidate_vbmm_55_ptr : RegionName := "candidate_vbmm_55"
  let candidate_vto_copy_58_ptr : RegionName := "candidate_vto_copy_58"
  let candidate_vsoftmax_backward_data_3_ptr : RegionName := "candidate_vsoftmax_backward_data_3"
  let candidate_vmasked_fill_12_ptr : RegionName := "candidate_vmasked_fill_12"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vbmm_56_ptr : RegionName := "candidate_vbmm_56"
  let candidate_vbmm_57_ptr : RegionName := "candidate_vbmm_57"
  let candidate_vslice_backward_21_ptr : RegionName := "candidate_vslice_backward_21"
  let candidate_vto_copy_59_ptr : RegionName := "candidate_vto_copy_59"
  let candidate_vslice_backward_22_ptr : RegionName := "candidate_vslice_backward_22"
  let candidate_vto_copy_60_ptr : RegionName := "candidate_vto_copy_60"
  let candidate_vslice_backward_23_ptr : RegionName := "candidate_vslice_backward_23"
  let candidate_vfill_6_ptr : RegionName := "candidate_vfill_6"
  let candidate_vslice_scatter_6_ptr : RegionName := "candidate_vslice_scatter_6"
  let candidate_vfill_7_ptr : RegionName := "candidate_vfill_7"
  let candidate_vslice_scatter_7_ptr : RegionName := "candidate_vslice_scatter_7"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vbmm_58_ptr : RegionName := "candidate_vbmm_58"
  let candidate_vbmm_59_ptr : RegionName := "candidate_vbmm_59"
  let candidate_vadd_26_ptr : RegionName := "candidate_vadd_26"
  let candidate_vselect_backward_3_ptr : RegionName := "candidate_vselect_backward_3"
  let candidate_vslice_backward_24_ptr : RegionName := "candidate_vslice_backward_24"
  let candidate_vadd_27_ptr : RegionName := "candidate_vadd_27"
  let candidate_vslice_backward_25_ptr : RegionName := "candidate_vslice_backward_25"
  let candidate_vadd_28_ptr : RegionName := "candidate_vadd_28"
  let candidate_vslice_backward_26_ptr : RegionName := "candidate_vslice_backward_26"
  let candidate_vadd_29_ptr : RegionName := "candidate_vadd_29"
  let candidate_vslice_backward_27_ptr : RegionName := "candidate_vslice_backward_27"
  let candidate_vadd_30_ptr : RegionName := "candidate_vadd_30"
  let candidate_vbmm_60_ptr : RegionName := "candidate_vbmm_60"
  let candidate_vbmm_61_ptr : RegionName := "candidate_vbmm_61"
  let candidate_vto_copy_61_ptr : RegionName := "candidate_vto_copy_61"
  let candidate_vsoftmax_backward_data_4_ptr : RegionName := "candidate_vsoftmax_backward_data_4"
  let candidate_vmasked_fill_13_ptr : RegionName := "candidate_vmasked_fill_13"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vbmm_62_ptr : RegionName := "candidate_vbmm_62"
  let candidate_vbmm_63_ptr : RegionName := "candidate_vbmm_63"
  let candidate_vslice_backward_28_ptr : RegionName := "candidate_vslice_backward_28"
  let candidate_vto_copy_62_ptr : RegionName := "candidate_vto_copy_62"
  let candidate_vslice_backward_29_ptr : RegionName := "candidate_vslice_backward_29"
  let candidate_vto_copy_63_ptr : RegionName := "candidate_vto_copy_63"
  let candidate_vslice_backward_30_ptr : RegionName := "candidate_vslice_backward_30"
  let candidate_vfill_8_ptr : RegionName := "candidate_vfill_8"
  let candidate_vslice_scatter_8_ptr : RegionName := "candidate_vslice_scatter_8"
  let candidate_vfill_9_ptr : RegionName := "candidate_vfill_9"
  let candidate_vslice_scatter_9_ptr : RegionName := "candidate_vslice_scatter_9"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vbmm_64_ptr : RegionName := "candidate_vbmm_64"
  let candidate_vbmm_65_ptr : RegionName := "candidate_vbmm_65"
  let candidate_vadd_31_ptr : RegionName := "candidate_vadd_31"
  let candidate_vselect_backward_4_ptr : RegionName := "candidate_vselect_backward_4"
  let candidate_vslice_backward_31_ptr : RegionName := "candidate_vslice_backward_31"
  let candidate_vadd_32_ptr : RegionName := "candidate_vadd_32"
  let candidate_vslice_backward_32_ptr : RegionName := "candidate_vslice_backward_32"
  let candidate_vadd_33_ptr : RegionName := "candidate_vadd_33"
  let candidate_vslice_backward_33_ptr : RegionName := "candidate_vslice_backward_33"
  let candidate_vadd_34_ptr : RegionName := "candidate_vadd_34"
  let candidate_vslice_backward_34_ptr : RegionName := "candidate_vslice_backward_34"
  let candidate_vadd_35_ptr : RegionName := "candidate_vadd_35"
  let candidate_vbmm_66_ptr : RegionName := "candidate_vbmm_66"
  let candidate_vbmm_67_ptr : RegionName := "candidate_vbmm_67"
  let candidate_vto_copy_64_ptr : RegionName := "candidate_vto_copy_64"
  let candidate_vsoftmax_backward_data_5_ptr : RegionName := "candidate_vsoftmax_backward_data_5"
  let candidate_vmasked_fill_14_ptr : RegionName := "candidate_vmasked_fill_14"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vbmm_68_ptr : RegionName := "candidate_vbmm_68"
  let candidate_vbmm_69_ptr : RegionName := "candidate_vbmm_69"
  let candidate_vslice_backward_35_ptr : RegionName := "candidate_vslice_backward_35"
  let candidate_vto_copy_65_ptr : RegionName := "candidate_vto_copy_65"
  let candidate_vslice_backward_36_ptr : RegionName := "candidate_vslice_backward_36"
  let candidate_vto_copy_66_ptr : RegionName := "candidate_vto_copy_66"
  let candidate_vslice_backward_37_ptr : RegionName := "candidate_vslice_backward_37"
  let candidate_vfill_10_ptr : RegionName := "candidate_vfill_10"
  let candidate_vslice_scatter_10_ptr : RegionName := "candidate_vslice_scatter_10"
  let candidate_vfill_11_ptr : RegionName := "candidate_vfill_11"
  let candidate_vslice_scatter_11_ptr : RegionName := "candidate_vslice_scatter_11"
  let candidate_vsum_6_ptr : RegionName := "candidate_vsum_6"
  let candidate_vbmm_70_ptr : RegionName := "candidate_vbmm_70"
  let candidate_vbmm_71_ptr : RegionName := "candidate_vbmm_71"
  let candidate_vadd_36_ptr : RegionName := "candidate_vadd_36"
  let candidate_vselect_backward_5_ptr : RegionName := "candidate_vselect_backward_5"
  let candidate_vslice_backward_38_ptr : RegionName := "candidate_vslice_backward_38"
  let candidate_vadd_37_ptr : RegionName := "candidate_vadd_37"
  let candidate_vslice_backward_39_ptr : RegionName := "candidate_vslice_backward_39"
  let candidate_vadd_38_ptr : RegionName := "candidate_vadd_38"
  let candidate_vslice_backward_40_ptr : RegionName := "candidate_vslice_backward_40"
  let candidate_vadd_39_ptr : RegionName := "candidate_vadd_39"
  let candidate_vslice_backward_41_ptr : RegionName := "candidate_vslice_backward_41"
  let candidate_vadd_40_ptr : RegionName := "candidate_vadd_40"
  let candidate_vbmm_72_ptr : RegionName := "candidate_vbmm_72"
  let candidate_vbmm_73_ptr : RegionName := "candidate_vbmm_73"
  let candidate_vto_copy_67_ptr : RegionName := "candidate_vto_copy_67"
  let candidate_vsoftmax_backward_data_6_ptr : RegionName := "candidate_vsoftmax_backward_data_6"
  let candidate_vmasked_fill_15_ptr : RegionName := "candidate_vmasked_fill_15"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vbmm_74_ptr : RegionName := "candidate_vbmm_74"
  let candidate_vbmm_75_ptr : RegionName := "candidate_vbmm_75"
  let candidate_vslice_backward_42_ptr : RegionName := "candidate_vslice_backward_42"
  let candidate_vto_copy_68_ptr : RegionName := "candidate_vto_copy_68"
  let candidate_vslice_backward_43_ptr : RegionName := "candidate_vslice_backward_43"
  let candidate_vto_copy_69_ptr : RegionName := "candidate_vto_copy_69"
  let candidate_vslice_backward_44_ptr : RegionName := "candidate_vslice_backward_44"
  let candidate_vfill_12_ptr : RegionName := "candidate_vfill_12"
  let candidate_vslice_scatter_12_ptr : RegionName := "candidate_vslice_scatter_12"
  let candidate_vfill_13_ptr : RegionName := "candidate_vfill_13"
  let candidate_vslice_scatter_13_ptr : RegionName := "candidate_vslice_scatter_13"
  let candidate_vsum_7_ptr : RegionName := "candidate_vsum_7"
  let candidate_vbmm_76_ptr : RegionName := "candidate_vbmm_76"
  let candidate_vbmm_77_ptr : RegionName := "candidate_vbmm_77"
  let candidate_vadd_41_ptr : RegionName := "candidate_vadd_41"
  let candidate_vselect_backward_6_ptr : RegionName := "candidate_vselect_backward_6"
  let candidate_vslice_backward_45_ptr : RegionName := "candidate_vslice_backward_45"
  let candidate_vadd_42_ptr : RegionName := "candidate_vadd_42"
  let candidate_vslice_backward_46_ptr : RegionName := "candidate_vslice_backward_46"
  let candidate_vadd_43_ptr : RegionName := "candidate_vadd_43"
  let candidate_vslice_backward_47_ptr : RegionName := "candidate_vslice_backward_47"
  let candidate_vadd_44_ptr : RegionName := "candidate_vadd_44"
  let candidate_vslice_backward_48_ptr : RegionName := "candidate_vslice_backward_48"
  let candidate_vadd_45_ptr : RegionName := "candidate_vadd_45"
  let candidate_vbmm_78_ptr : RegionName := "candidate_vbmm_78"
  let candidate_vbmm_79_ptr : RegionName := "candidate_vbmm_79"
  let candidate_vto_copy_70_ptr : RegionName := "candidate_vto_copy_70"
  let candidate_vsoftmax_backward_data_7_ptr : RegionName := "candidate_vsoftmax_backward_data_7"
  let candidate_vmasked_fill_16_ptr : RegionName := "candidate_vmasked_fill_16"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vbmm_80_ptr : RegionName := "candidate_vbmm_80"
  let candidate_vbmm_81_ptr : RegionName := "candidate_vbmm_81"
  let candidate_vslice_backward_49_ptr : RegionName := "candidate_vslice_backward_49"
  let candidate_vto_copy_71_ptr : RegionName := "candidate_vto_copy_71"
  let candidate_vslice_backward_50_ptr : RegionName := "candidate_vslice_backward_50"
  let candidate_vto_copy_72_ptr : RegionName := "candidate_vto_copy_72"
  let candidate_vslice_backward_51_ptr : RegionName := "candidate_vslice_backward_51"
  let candidate_vfill_14_ptr : RegionName := "candidate_vfill_14"
  let candidate_vslice_scatter_14_ptr : RegionName := "candidate_vslice_scatter_14"
  let candidate_vfill_15_ptr : RegionName := "candidate_vfill_15"
  let candidate_vslice_scatter_15_ptr : RegionName := "candidate_vslice_scatter_15"
  let candidate_vsum_8_ptr : RegionName := "candidate_vsum_8"
  let candidate_vbmm_82_ptr : RegionName := "candidate_vbmm_82"
  let candidate_vbmm_83_ptr : RegionName := "candidate_vbmm_83"
  let candidate_vadd_46_ptr : RegionName := "candidate_vadd_46"
  let candidate_vselect_backward_7_ptr : RegionName := "candidate_vselect_backward_7"
  let candidate_vslice_backward_52_ptr : RegionName := "candidate_vslice_backward_52"
  let candidate_vadd_47_ptr : RegionName := "candidate_vadd_47"
  let candidate_vslice_backward_53_ptr : RegionName := "candidate_vslice_backward_53"
  let candidate_vadd_48_ptr : RegionName := "candidate_vadd_48"
  let candidate_vslice_backward_54_ptr : RegionName := "candidate_vslice_backward_54"
  let candidate_vadd_49_ptr : RegionName := "candidate_vadd_49"
  let candidate_vslice_backward_55_ptr : RegionName := "candidate_vslice_backward_55"
  let candidate_vadd_50_ptr : RegionName := "candidate_vadd_50"
  let candidate_vbmm_84_ptr : RegionName := "candidate_vbmm_84"
  let candidate_vbmm_85_ptr : RegionName := "candidate_vbmm_85"
  let candidate_vto_copy_73_ptr : RegionName := "candidate_vto_copy_73"
  let candidate_vsoftmax_backward_data_8_ptr : RegionName := "candidate_vsoftmax_backward_data_8"
  let candidate_vmasked_fill_17_ptr : RegionName := "candidate_vmasked_fill_17"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vbmm_86_ptr : RegionName := "candidate_vbmm_86"
  let candidate_vbmm_87_ptr : RegionName := "candidate_vbmm_87"
  let candidate_vslice_backward_56_ptr : RegionName := "candidate_vslice_backward_56"
  let candidate_vto_copy_74_ptr : RegionName := "candidate_vto_copy_74"
  let candidate_vslice_backward_57_ptr : RegionName := "candidate_vslice_backward_57"
  let candidate_vto_copy_75_ptr : RegionName := "candidate_vto_copy_75"
  let candidate_vslice_backward_58_ptr : RegionName := "candidate_vslice_backward_58"
  let candidate_vfill_16_ptr : RegionName := "candidate_vfill_16"
  let candidate_vslice_scatter_16_ptr : RegionName := "candidate_vslice_scatter_16"
  let candidate_vfill_17_ptr : RegionName := "candidate_vfill_17"
  let candidate_vslice_scatter_17_ptr : RegionName := "candidate_vslice_scatter_17"
  let candidate_vsum_9_ptr : RegionName := "candidate_vsum_9"
  let candidate_vbmm_88_ptr : RegionName := "candidate_vbmm_88"
  let candidate_vbmm_89_ptr : RegionName := "candidate_vbmm_89"
  let candidate_vadd_51_ptr : RegionName := "candidate_vadd_51"
  let candidate_vselect_backward_8_ptr : RegionName := "candidate_vselect_backward_8"
  let candidate_vslice_backward_59_ptr : RegionName := "candidate_vslice_backward_59"
  let candidate_vadd_52_ptr : RegionName := "candidate_vadd_52"
  let candidate_vslice_backward_60_ptr : RegionName := "candidate_vslice_backward_60"
  let candidate_vadd_53_ptr : RegionName := "candidate_vadd_53"
  let candidate_vslice_backward_61_ptr : RegionName := "candidate_vslice_backward_61"
  let candidate_vadd_54_ptr : RegionName := "candidate_vadd_54"
  let candidate_vslice_backward_62_ptr : RegionName := "candidate_vslice_backward_62"
  let candidate_vadd_55_ptr : RegionName := "candidate_vadd_55"
  let candidate_vcat_36_ptr : RegionName := "candidate_vcat_36"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vadd_56_ptr : RegionName := "candidate_vadd_56"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vadd_57_ptr : RegionName := "candidate_vadd_57"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vadd_58_ptr : RegionName := "candidate_vadd_58"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vadd_59_ptr : RegionName := "candidate_vadd_59"
  let candidate_vslice_backward_63_ptr : RegionName := "candidate_vslice_backward_63"
  let candidate_vslice_backward_64_ptr : RegionName := "candidate_vslice_backward_64"
  let candidate_vadd_60_ptr : RegionName := "candidate_vadd_60"
  let candidate_vslice_backward_65_ptr : RegionName := "candidate_vslice_backward_65"
  let candidate_vslice_backward_66_ptr : RegionName := "candidate_vslice_backward_66"
  let candidate_vadd_61_ptr : RegionName := "candidate_vadd_61"
  let candidate_vcat_37_ptr : RegionName := "candidate_vcat_37"
  let candidate_vto_copy_76_ptr : RegionName := "candidate_vto_copy_76"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vsum_10_ptr : RegionName := "candidate_vsum_10"
  let candidate_vpow_4_ptr : RegionName := "candidate_vpow_4"
  let candidate_vmul_50_ptr : RegionName := "candidate_vmul_50"
  let candidate_vmul_51_ptr : RegionName := "candidate_vmul_51"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_vpow_5_ptr : RegionName := "candidate_vpow_5"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vadd_62_ptr : RegionName := "candidate_vadd_62"
  let candidate_vto_copy_77_ptr : RegionName := "candidate_vto_copy_77"
  let candidate_vcat_38_ptr : RegionName := "candidate_vcat_38"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vto_copy_78_ptr : RegionName := "candidate_vto_copy_78"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vmul_56_ptr : RegionName := "candidate_vmul_56"
  let candidate_vsum_11_ptr : RegionName := "candidate_vsum_11"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_57_ptr : RegionName := "candidate_vmul_57"
  let candidate_vmul_58_ptr : RegionName := "candidate_vmul_58"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vpow_7_ptr : RegionName := "candidate_vpow_7"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vadd_63_ptr : RegionName := "candidate_vadd_63"
  let candidate_vto_copy_79_ptr : RegionName := "candidate_vto_copy_79"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vadd_64_ptr : RegionName := "candidate_vadd_64"
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
    r78_row = tl.arange($(96))
    r78_col = tl.arange($(1024))
    r78_lane = tl.arange($(512))
    for r78_group in range($(1)) {
      r78_left = tl.load($(candidate_vcat_7_ptr) + ((((((r78_row[:, None] * $(512)) + r78_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r78_row[:, None] * $(512)) + r78_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r78_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r78_lane[:, None] * $(1024)) + r78_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r78_lane[:, None] * $(1024)) + r78_col[None, :]) % $(128)) * $(512)) + ((((r78_lane[:, None] * $(1024)) + r78_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r78_value = tl.dot(r78_left, r78_right)
      r78_value = r78_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_4_ptr) + ((r78_group * $(98304)) + (r78_row[:, None] * $(1024)) + r78_col[None, :]), r78_value)
    }
    r79_row = tl.arange($(96))
    r79_col = tl.arange($(1024))
    r79_lane = tl.arange($(512))
    for r79_group in range($(1)) {
      r79_left = tl.load($(candidate_vcat_7_ptr) + ((((((r79_row[:, None] * $(512)) + r79_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r79_row[:, None] * $(512)) + r79_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r79_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r79_lane[:, None] * $(1024)) + r79_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r79_lane[:, None] * $(1024)) + r79_col[None, :]) % $(128)) * $(512)) + ((((r79_lane[:, None] * $(1024)) + r79_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r79_value = tl.dot(r79_left, r79_right)
      r79_value = r79_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r79_group * $(98304)) + (r79_row[:, None] * $(1024)) + r79_col[None, :]), r79_value)
    }
    r80_z = tl.arange($(49152))
    r80_value = tl.where((r80_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r80_z / $(1536)) % $(32)) * $(1024)) + (((r80_z / $(192)) % $(8)) * $(128)) + ((r80_z % $(192)) - $(0)) + $(32768)) / $(1024)) % $(288)) * $(1536)) + (((((((r80_z / $(1536)) % $(32)) * $(1024)) + (((r80_z / $(192)) % $(8)) * $(128)) + ((r80_z % $(192)) - $(0)) + $(32768)) / $(128)) % $(8)) * $(192)) + (((((r80_z / $(1536)) % $(32)) * $(1024)) + (((r80_z / $(192)) % $(8)) * $(128)) + ((r80_z % $(192)) - $(0)) + $(32768)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r80_z / $(1536)) % $(32)) * $(512)) + (((r80_z / $(192)) % $(8)) * $(64)) + ((r80_z % $(192)) - $(128)) + $(16384)) / $(64)) % $(8)) * $(18432)) + (((((((r80_z / $(1536)) % $(32)) * $(512)) + (((r80_z / $(192)) % $(8)) * $(64)) + ((r80_z % $(192)) - $(128)) + $(16384)) / $(512)) % $(288)) * $(64)) + (((((r80_z / $(1536)) % $(32)) * $(512)) + (((r80_z / $(192)) % $(8)) * $(64)) + ((r80_z % $(192)) - $(128)) + $(16384)) % $(64))), dtype=tl.bfloat16))
    r80_value = r80_value.to(tl.bfloat16)
    r80_value = tl.broadcast_to(r80_value, [$(49152)])
    tl.store($(candidate_vcat_11_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(147456))
    r81_value = tl.where((r81_z % $(192)) < $(128), tl.load($(candidate_vbmm_4_ptr) + ((((((((r81_z / $(1536)) % $(96)) * $(1024)) + (((r81_z / $(192)) % $(8)) * $(128)) + ((r81_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r81_z / $(1536)) % $(96)) * $(1024)) + (((r81_z / $(192)) % $(8)) * $(128)) + ((r81_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r81_z / $(1536)) % $(96)) * $(1024)) + (((r81_z / $(192)) % $(8)) * $(128)) + ((r81_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r81_z / $(1536)) % $(96)) * $(576)) + ((((r81_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r81_value = r81_value.to(tl.bfloat16)
    r81_value = tl.broadcast_to(r81_value, [$(147456)])
    tl.store($(candidate_vcat_12_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(49152))
    r82_value = tl.load($(candidate_vcat_11_ptr) + ((((r82_z / $(192)) % $(32)) * $(1536)) + (((r82_z / $(6144)) % $(8)) * $(192)) + (r82_z % $(192))), dtype=tl.bfloat16)
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(49152)])
    tl.store($(candidate_vto_copy_25_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(98304))
    r83_value = tl.load($(candidate_vcat_12_ptr) + ((((r83_z / $(192)) % $(64)) * $(1536)) + (((r83_z / $(12288)) % $(8)) * $(192)) + (r83_z % $(192))), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(98304)])
    tl.store($(candidate_vto_copy_26_ptr) + r83_z, r83_value)
    r84_row = tl.arange($(32))
    r84_col = tl.arange($(64))
    r84_lane = tl.arange($(192))
    for r84_group in range($(8)) {
      r84_left = tl.load($(candidate_vto_copy_25_ptr) + (((((((r84_group % $(8)) * $(6144)) + (r84_row[:, None] * $(192)) + r84_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r84_group % $(8)) * $(6144)) + (r84_row[:, None] * $(192)) + r84_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r84_group % $(8)) * $(6144)) + (r84_row[:, None] * $(192)) + r84_lane[None, :]) % $(192))), dtype=tl.float32)
      r84_right = tl.load($(candidate_vto_copy_26_ptr) + (((((((r84_group % $(8)) * $(12288)) + (r84_lane[:, None] * $(64)) + r84_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r84_group % $(8)) * $(12288)) + (r84_lane[:, None] * $(64)) + r84_col[None, :]) % $(64)) * $(192)) + (((((r84_group % $(8)) * $(12288)) + (r84_lane[:, None] * $(64)) + r84_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r84_value = tl.dot(r84_left, r84_right)
      r84_value = r84_value.to(tl.float32)
      tl.store($(candidate_vbmm_6_ptr) + ((r84_group * $(2048)) + (r84_row[:, None] * $(64)) + r84_col[None, :]), r84_value)
    }
    r85_z = tl.arange($(16384))
    r85_value = (tl.load($(candidate_vbmm_6_ptr) + ((((r85_z / $(2048)) % $(8)) * $(2048)) + (((r85_z / $(64)) % $(32)) * $(64)) + (r85_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(16384)])
    tl.store($(candidate_vmul_22_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(32))
    r86_value = (tl.cast(r86_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r86_value = tl.broadcast_to(r86_value, [$(32)])
    tl.store($(candidate_vadd_7_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(32))
    r87_value = (tl.load($(candidate_vadd_7_ptr) + r87_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r87_value = tl.broadcast_to(r87_value, [$(32)])
    tl.store($(candidate_vsub_16_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(2048))
    r88_value = (tl.cast((r88_z % $(64)), tl.int64) > tl.load($(candidate_vsub_16_ptr) + ((r88_z / $(64)) % $(32)), dtype=tl.int64))
    r88_value = tl.broadcast_to(r88_value, [$(2048)])
    tl.store($(candidate_vgt_1_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(16384))
    r89_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r89_z / $(64)) % $(32)) * $(64)) + (r89_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_22_ptr) + ((((r89_z / $(2048)) % $(8)) * $(2048)) + (((r89_z / $(64)) % $(32)) * $(64)) + (r89_z % $(64))), dtype=tl.float32))
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(16384))
    r90_lane = tl.arange($(64))
    r90_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r90_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r90_z[:, None] / $(64)) % $(32)) * $(64)) + (r90_lane[None, :] % $(64))), dtype=tl.float32)
    r90_maximum = tl.max(r90_row_values, axis=1, keep_dims=true)
    r90_weights = tl.exp(r90_row_values - r90_maximum)
    r90_total = tl.sum(r90_weights, axis=1)
    r90_maximum = tl.max(r90_row_values, axis=1)
    r90_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r90_z / $(2048)) % $(8)) * $(2048)) + (((r90_z / $(64)) % $(32)) * $(64)) + (r90_z % $(64))), dtype=tl.float32) - r90_maximum) / r90_total
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(16384)])
    tl.store($(candidate_vsoftmax_1_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(16384))
    r91_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r91_z / $(2048)) % $(8)) * $(2048)) + (((r91_z / $(64)) % $(32)) * $(64)) + (r91_z % $(64))), dtype=tl.float32)
    r91_value = r91_value.to(tl.bfloat16)
    r91_value = tl.broadcast_to(r91_value, [$(16384)])
    tl.store($(candidate_vto_copy_27_ptr) + r91_z, r91_value)
    r92_row = tl.arange($(96))
    r92_col = tl.arange($(1024))
    r92_lane = tl.arange($(512))
    for r92_group in range($(1)) {
      r92_left = tl.load($(candidate_vcat_7_ptr) + ((((((r92_row[:, None] * $(512)) + r92_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r92_row[:, None] * $(512)) + r92_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r92_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r92_lane[:, None] * $(1024)) + r92_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r92_lane[:, None] * $(1024)) + r92_col[None, :]) % $(128)) * $(512)) + ((((r92_lane[:, None] * $(1024)) + r92_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r92_value = tl.dot(r92_left, r92_right)
      r92_value = r92_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_8_ptr) + ((r92_group * $(98304)) + (r92_row[:, None] * $(1024)) + r92_col[None, :]), r92_value)
    }
    r93_row = tl.arange($(96))
    r93_col = tl.arange($(1024))
    r93_lane = tl.arange($(512))
    for r93_group in range($(1)) {
      r93_left = tl.load($(candidate_vcat_7_ptr) + ((((((r93_row[:, None] * $(512)) + r93_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r93_row[:, None] * $(512)) + r93_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r93_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r93_lane[:, None] * $(1024)) + r93_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r93_lane[:, None] * $(1024)) + r93_col[None, :]) % $(128)) * $(512)) + ((((r93_lane[:, None] * $(1024)) + r93_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r93_value = tl.dot(r93_left, r93_right)
      r93_value = r93_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_9_ptr) + ((r93_group * $(98304)) + (r93_row[:, None] * $(1024)) + r93_col[None, :]), r93_value)
    }
    r94_z = tl.arange($(49152))
    r94_value = tl.where((r94_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r94_z / $(1536)) % $(32)) * $(1024)) + (((r94_z / $(192)) % $(8)) * $(128)) + ((r94_z % $(192)) - $(0)) + $(65536)) / $(1024)) % $(288)) * $(1536)) + (((((((r94_z / $(1536)) % $(32)) * $(1024)) + (((r94_z / $(192)) % $(8)) * $(128)) + ((r94_z % $(192)) - $(0)) + $(65536)) / $(128)) % $(8)) * $(192)) + (((((r94_z / $(1536)) % $(32)) * $(1024)) + (((r94_z / $(192)) % $(8)) * $(128)) + ((r94_z % $(192)) - $(0)) + $(65536)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r94_z / $(1536)) % $(32)) * $(512)) + (((r94_z / $(192)) % $(8)) * $(64)) + ((r94_z % $(192)) - $(128)) + $(32768)) / $(64)) % $(8)) * $(18432)) + (((((((r94_z / $(1536)) % $(32)) * $(512)) + (((r94_z / $(192)) % $(8)) * $(64)) + ((r94_z % $(192)) - $(128)) + $(32768)) / $(512)) % $(288)) * $(64)) + (((((r94_z / $(1536)) % $(32)) * $(512)) + (((r94_z / $(192)) % $(8)) * $(64)) + ((r94_z % $(192)) - $(128)) + $(32768)) % $(64))), dtype=tl.bfloat16))
    r94_value = r94_value.to(tl.bfloat16)
    r94_value = tl.broadcast_to(r94_value, [$(49152)])
    tl.store($(candidate_vcat_14_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(147456))
    r95_value = tl.where((r95_z % $(192)) < $(128), tl.load($(candidate_vbmm_8_ptr) + ((((((((r95_z / $(1536)) % $(96)) * $(1024)) + (((r95_z / $(192)) % $(8)) * $(128)) + ((r95_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r95_z / $(1536)) % $(96)) * $(1024)) + (((r95_z / $(192)) % $(8)) * $(128)) + ((r95_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r95_z / $(1536)) % $(96)) * $(1024)) + (((r95_z / $(192)) % $(8)) * $(128)) + ((r95_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r95_z / $(1536)) % $(96)) * $(576)) + ((((r95_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r95_value = r95_value.to(tl.bfloat16)
    r95_value = tl.broadcast_to(r95_value, [$(147456)])
    tl.store($(candidate_vcat_15_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(49152))
    r96_value = tl.load($(candidate_vcat_14_ptr) + ((((r96_z / $(192)) % $(32)) * $(1536)) + (((r96_z / $(6144)) % $(8)) * $(192)) + (r96_z % $(192))), dtype=tl.bfloat16)
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(49152)])
    tl.store($(candidate_vto_copy_28_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(147456))
    r97_value = tl.load($(candidate_vcat_15_ptr) + ((((r97_z / $(192)) % $(96)) * $(1536)) + (((r97_z / $(18432)) % $(8)) * $(192)) + (r97_z % $(192))), dtype=tl.bfloat16)
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(147456)])
    tl.store($(candidate_vto_copy_29_ptr) + r97_z, r97_value)
    r98_row = tl.arange($(32))
    r98_col = tl.arange($(96))
    r98_lane = tl.arange($(192))
    for r98_group in range($(8)) {
      r98_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r98_group % $(8)) * $(6144)) + (r98_row[:, None] * $(192)) + r98_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r98_group % $(8)) * $(6144)) + (r98_row[:, None] * $(192)) + r98_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r98_group % $(8)) * $(6144)) + (r98_row[:, None] * $(192)) + r98_lane[None, :]) % $(192))), dtype=tl.float32)
      r98_right = tl.load($(candidate_vto_copy_29_ptr) + (((((((r98_group % $(8)) * $(18432)) + (r98_lane[:, None] * $(96)) + r98_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r98_group % $(8)) * $(18432)) + (r98_lane[:, None] * $(96)) + r98_col[None, :]) % $(96)) * $(192)) + (((((r98_group % $(8)) * $(18432)) + (r98_lane[:, None] * $(96)) + r98_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r98_value = tl.dot(r98_left, r98_right)
      r98_value = r98_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r98_group * $(3072)) + (r98_row[:, None] * $(96)) + r98_col[None, :]), r98_value)
    }
    r99_z = tl.arange($(24576))
    r99_value = (tl.load($(candidate_vbmm_10_ptr) + ((((r99_z / $(3072)) % $(8)) * $(3072)) + (((r99_z / $(96)) % $(32)) * $(96)) + (r99_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(24576)])
    tl.store($(candidate_vmul_23_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(32))
    r100_value = (tl.cast(r100_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r100_value = tl.broadcast_to(r100_value, [$(32)])
    tl.store($(candidate_vadd_8_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(32))
    r101_value = (tl.load($(candidate_vadd_8_ptr) + r101_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r101_value = tl.broadcast_to(r101_value, [$(32)])
    tl.store($(candidate_vsub_17_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(3072))
    r102_value = (tl.cast((r102_z % $(96)), tl.int64) > tl.load($(candidate_vsub_17_ptr) + ((r102_z / $(96)) % $(32)), dtype=tl.int64))
    r102_value = tl.broadcast_to(r102_value, [$(3072)])
    tl.store($(candidate_vgt_2_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(24576))
    r103_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r103_z / $(96)) % $(32)) * $(96)) + (r103_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r103_z / $(3072)) % $(8)) * $(3072)) + (((r103_z / $(96)) % $(32)) * $(96)) + (r103_z % $(96))), dtype=tl.float32))
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(24576))
    r104_lane = tl.arange($(96))
    r104_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r104_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r104_z[:, None] / $(96)) % $(32)) * $(96)) + (r104_lane[None, :] % $(96))), dtype=tl.float32)
    r104_maximum = tl.max(r104_row_values, axis=1, keep_dims=true)
    r104_weights = tl.exp(r104_row_values - r104_maximum)
    r104_total = tl.sum(r104_weights, axis=1)
    r104_maximum = tl.max(r104_row_values, axis=1)
    r104_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r104_z / $(3072)) % $(8)) * $(3072)) + (((r104_z / $(96)) % $(32)) * $(96)) + (r104_z % $(96))), dtype=tl.float32) - r104_maximum) / r104_total
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(24576)])
    tl.store($(candidate_vsoftmax_2_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(24576))
    r105_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r105_z / $(3072)) % $(8)) * $(3072)) + (((r105_z / $(96)) % $(32)) * $(96)) + (r105_z % $(96))), dtype=tl.float32)
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(24576)])
    tl.store($(candidate_vto_copy_30_ptr) + r105_z, r105_value)
    r106_row = tl.arange($(96))
    r106_col = tl.arange($(1024))
    r106_lane = tl.arange($(512))
    for r106_group in range($(1)) {
      r106_left = tl.load($(candidate_vcat_7_ptr) + ((((((r106_row[:, None] * $(512)) + r106_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r106_row[:, None] * $(512)) + r106_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r106_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r106_lane[:, None] * $(1024)) + r106_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r106_lane[:, None] * $(1024)) + r106_col[None, :]) % $(128)) * $(512)) + ((((r106_lane[:, None] * $(1024)) + r106_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r106_value = tl.dot(r106_left, r106_right)
      r106_value = r106_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r106_group * $(98304)) + (r106_row[:, None] * $(1024)) + r106_col[None, :]), r106_value)
    }
    r107_row = tl.arange($(96))
    r107_col = tl.arange($(1024))
    r107_lane = tl.arange($(512))
    for r107_group in range($(1)) {
      r107_left = tl.load($(candidate_vcat_7_ptr) + ((((((r107_row[:, None] * $(512)) + r107_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r107_row[:, None] * $(512)) + r107_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r107_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r107_lane[:, None] * $(1024)) + r107_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r107_lane[:, None] * $(1024)) + r107_col[None, :]) % $(128)) * $(512)) + ((((r107_lane[:, None] * $(1024)) + r107_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r107_value = tl.dot(r107_left, r107_right)
      r107_value = r107_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_13_ptr) + ((r107_group * $(98304)) + (r107_row[:, None] * $(1024)) + r107_col[None, :]), r107_value)
    }
    r108_z = tl.arange($(49152))
    r108_value = tl.where((r108_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r108_z / $(1536)) % $(32)) * $(1024)) + (((r108_z / $(192)) % $(8)) * $(128)) + ((r108_z % $(192)) - $(0)) + $(98304)) / $(1024)) % $(288)) * $(1536)) + (((((((r108_z / $(1536)) % $(32)) * $(1024)) + (((r108_z / $(192)) % $(8)) * $(128)) + ((r108_z % $(192)) - $(0)) + $(98304)) / $(128)) % $(8)) * $(192)) + (((((r108_z / $(1536)) % $(32)) * $(1024)) + (((r108_z / $(192)) % $(8)) * $(128)) + ((r108_z % $(192)) - $(0)) + $(98304)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r108_z / $(1536)) % $(32)) * $(512)) + (((r108_z / $(192)) % $(8)) * $(64)) + ((r108_z % $(192)) - $(128)) + $(49152)) / $(64)) % $(8)) * $(18432)) + (((((((r108_z / $(1536)) % $(32)) * $(512)) + (((r108_z / $(192)) % $(8)) * $(64)) + ((r108_z % $(192)) - $(128)) + $(49152)) / $(512)) % $(288)) * $(64)) + (((((r108_z / $(1536)) % $(32)) * $(512)) + (((r108_z / $(192)) % $(8)) * $(64)) + ((r108_z % $(192)) - $(128)) + $(49152)) % $(64))), dtype=tl.bfloat16))
    r108_value = r108_value.to(tl.bfloat16)
    r108_value = tl.broadcast_to(r108_value, [$(49152)])
    tl.store($(candidate_vcat_17_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(147456))
    r109_value = tl.where((r109_z % $(192)) < $(128), tl.load($(candidate_vbmm_12_ptr) + ((((((((r109_z / $(1536)) % $(96)) * $(1024)) + (((r109_z / $(192)) % $(8)) * $(128)) + ((r109_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r109_z / $(1536)) % $(96)) * $(1024)) + (((r109_z / $(192)) % $(8)) * $(128)) + ((r109_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r109_z / $(1536)) % $(96)) * $(1024)) + (((r109_z / $(192)) % $(8)) * $(128)) + ((r109_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r109_z / $(1536)) % $(96)) * $(576)) + ((((r109_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r109_value = r109_value.to(tl.bfloat16)
    r109_value = tl.broadcast_to(r109_value, [$(147456)])
    tl.store($(candidate_vcat_18_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(49152))
    r110_value = tl.load($(candidate_vcat_17_ptr) + ((((r110_z / $(192)) % $(32)) * $(1536)) + (((r110_z / $(6144)) % $(8)) * $(192)) + (r110_z % $(192))), dtype=tl.bfloat16)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(49152)])
    tl.store($(candidate_vto_copy_31_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(49152))
    r111_value = tl.load($(candidate_vcat_18_ptr) + ((((r111_z / $(192)) % $(32)) * $(1536)) + (((r111_z / $(6144)) % $(8)) * $(192)) + (r111_z % $(192))), dtype=tl.bfloat16)
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(49152)])
    tl.store($(candidate_vto_copy_32_ptr) + r111_z, r111_value)
    r112_row = tl.arange($(32))
    r112_col = tl.arange($(32))
    r112_lane = tl.arange($(192))
    for r112_group in range($(8)) {
      r112_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r112_group % $(8)) * $(6144)) + (r112_row[:, None] * $(192)) + r112_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r112_group % $(8)) * $(6144)) + (r112_row[:, None] * $(192)) + r112_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r112_group % $(8)) * $(6144)) + (r112_row[:, None] * $(192)) + r112_lane[None, :]) % $(192))), dtype=tl.float32)
      r112_right = tl.load($(candidate_vto_copy_32_ptr) + (((((((r112_group % $(8)) * $(6144)) + (r112_lane[:, None] * $(32)) + r112_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r112_group % $(8)) * $(6144)) + (r112_lane[:, None] * $(32)) + r112_col[None, :]) % $(32)) * $(192)) + (((((r112_group % $(8)) * $(6144)) + (r112_lane[:, None] * $(32)) + r112_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r112_value = tl.dot(r112_left, r112_right)
      r112_value = r112_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r112_group * $(1024)) + (r112_row[:, None] * $(32)) + r112_col[None, :]), r112_value)
    }
    r113_z = tl.arange($(8192))
    r113_value = (tl.load($(candidate_vbmm_14_ptr) + ((((r113_z / $(1024)) % $(8)) * $(1024)) + (((r113_z / $(32)) % $(32)) * $(32)) + (r113_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(8192)])
    tl.store($(candidate_vmul_24_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(32))
    r114_value = (tl.cast(r114_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r114_value = tl.broadcast_to(r114_value, [$(32)])
    tl.store($(candidate_vadd_9_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(32))
    r115_value = (tl.load($(candidate_vadd_9_ptr) + r115_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r115_value = tl.broadcast_to(r115_value, [$(32)])
    tl.store($(candidate_vsub_18_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(1024))
    r116_value = (tl.cast((r116_z % $(32)), tl.int64) > tl.load($(candidate_vsub_18_ptr) + ((r116_z / $(32)) % $(32)), dtype=tl.int64))
    r116_value = tl.broadcast_to(r116_value, [$(1024)])
    tl.store($(candidate_vgt_3_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(8192))
    r117_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r117_z / $(32)) % $(32)) * $(32)) + (r117_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r117_z / $(1024)) % $(8)) * $(1024)) + (((r117_z / $(32)) % $(32)) * $(32)) + (r117_z % $(32))), dtype=tl.float32))
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(8192))
    r118_lane = tl.arange($(32))
    r118_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r118_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r118_z[:, None] / $(32)) % $(32)) * $(32)) + (r118_lane[None, :] % $(32))), dtype=tl.float32)
    r118_maximum = tl.max(r118_row_values, axis=1, keep_dims=true)
    r118_weights = tl.exp(r118_row_values - r118_maximum)
    r118_total = tl.sum(r118_weights, axis=1)
    r118_maximum = tl.max(r118_row_values, axis=1)
    r118_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r118_z / $(1024)) % $(8)) * $(1024)) + (((r118_z / $(32)) % $(32)) * $(32)) + (r118_z % $(32))), dtype=tl.float32) - r118_maximum) / r118_total
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(8192)])
    tl.store($(candidate_vsoftmax_3_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(8192))
    r119_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r119_z / $(1024)) % $(8)) * $(1024)) + (((r119_z / $(32)) % $(32)) * $(32)) + (r119_z % $(32))), dtype=tl.float32)
    r119_value = r119_value.to(tl.bfloat16)
    r119_value = tl.broadcast_to(r119_value, [$(8192)])
    tl.store($(candidate_vto_copy_33_ptr) + r119_z, r119_value)
    r120_row = tl.arange($(96))
    r120_col = tl.arange($(1024))
    r120_lane = tl.arange($(512))
    for r120_group in range($(1)) {
      r120_left = tl.load($(candidate_vcat_7_ptr) + ((((((r120_row[:, None] * $(512)) + r120_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r120_row[:, None] * $(512)) + r120_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r120_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r120_lane[:, None] * $(1024)) + r120_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r120_lane[:, None] * $(1024)) + r120_col[None, :]) % $(128)) * $(512)) + ((((r120_lane[:, None] * $(1024)) + r120_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r120_value = tl.dot(r120_left, r120_right)
      r120_value = r120_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_16_ptr) + ((r120_group * $(98304)) + (r120_row[:, None] * $(1024)) + r120_col[None, :]), r120_value)
    }
    r121_row = tl.arange($(96))
    r121_col = tl.arange($(1024))
    r121_lane = tl.arange($(512))
    for r121_group in range($(1)) {
      r121_left = tl.load($(candidate_vcat_7_ptr) + ((((((r121_row[:, None] * $(512)) + r121_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r121_row[:, None] * $(512)) + r121_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r121_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r121_lane[:, None] * $(1024)) + r121_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r121_lane[:, None] * $(1024)) + r121_col[None, :]) % $(128)) * $(512)) + ((((r121_lane[:, None] * $(1024)) + r121_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r121_value = tl.dot(r121_left, r121_right)
      r121_value = r121_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_17_ptr) + ((r121_group * $(98304)) + (r121_row[:, None] * $(1024)) + r121_col[None, :]), r121_value)
    }
    r122_z = tl.arange($(49152))
    r122_value = tl.where((r122_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r122_z / $(1536)) % $(32)) * $(1024)) + (((r122_z / $(192)) % $(8)) * $(128)) + ((r122_z % $(192)) - $(0)) + $(131072)) / $(1024)) % $(288)) * $(1536)) + (((((((r122_z / $(1536)) % $(32)) * $(1024)) + (((r122_z / $(192)) % $(8)) * $(128)) + ((r122_z % $(192)) - $(0)) + $(131072)) / $(128)) % $(8)) * $(192)) + (((((r122_z / $(1536)) % $(32)) * $(1024)) + (((r122_z / $(192)) % $(8)) * $(128)) + ((r122_z % $(192)) - $(0)) + $(131072)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r122_z / $(1536)) % $(32)) * $(512)) + (((r122_z / $(192)) % $(8)) * $(64)) + ((r122_z % $(192)) - $(128)) + $(65536)) / $(64)) % $(8)) * $(18432)) + (((((((r122_z / $(1536)) % $(32)) * $(512)) + (((r122_z / $(192)) % $(8)) * $(64)) + ((r122_z % $(192)) - $(128)) + $(65536)) / $(512)) % $(288)) * $(64)) + (((((r122_z / $(1536)) % $(32)) * $(512)) + (((r122_z / $(192)) % $(8)) * $(64)) + ((r122_z % $(192)) - $(128)) + $(65536)) % $(64))), dtype=tl.bfloat16))
    r122_value = r122_value.to(tl.bfloat16)
    r122_value = tl.broadcast_to(r122_value, [$(49152)])
    tl.store($(candidate_vcat_20_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(147456))
    r123_value = tl.where((r123_z % $(192)) < $(128), tl.load($(candidate_vbmm_16_ptr) + ((((((((r123_z / $(1536)) % $(96)) * $(1024)) + (((r123_z / $(192)) % $(8)) * $(128)) + ((r123_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r123_z / $(1536)) % $(96)) * $(1024)) + (((r123_z / $(192)) % $(8)) * $(128)) + ((r123_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r123_z / $(1536)) % $(96)) * $(1024)) + (((r123_z / $(192)) % $(8)) * $(128)) + ((r123_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r123_z / $(1536)) % $(96)) * $(576)) + ((((r123_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r123_value = r123_value.to(tl.bfloat16)
    r123_value = tl.broadcast_to(r123_value, [$(147456)])
    tl.store($(candidate_vcat_21_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(49152))
    r124_value = tl.load($(candidate_vcat_20_ptr) + ((((r124_z / $(192)) % $(32)) * $(1536)) + (((r124_z / $(6144)) % $(8)) * $(192)) + (r124_z % $(192))), dtype=tl.bfloat16)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(49152)])
    tl.store($(candidate_vto_copy_34_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(98304))
    r125_value = tl.load($(candidate_vcat_21_ptr) + ((((r125_z / $(192)) % $(64)) * $(1536)) + (((r125_z / $(12288)) % $(8)) * $(192)) + (r125_z % $(192))), dtype=tl.bfloat16)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(98304)])
    tl.store($(candidate_vto_copy_35_ptr) + r125_z, r125_value)
    r126_row = tl.arange($(32))
    r126_col = tl.arange($(64))
    r126_lane = tl.arange($(192))
    for r126_group in range($(8)) {
      r126_left = tl.load($(candidate_vto_copy_34_ptr) + (((((((r126_group % $(8)) * $(6144)) + (r126_row[:, None] * $(192)) + r126_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r126_group % $(8)) * $(6144)) + (r126_row[:, None] * $(192)) + r126_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r126_group % $(8)) * $(6144)) + (r126_row[:, None] * $(192)) + r126_lane[None, :]) % $(192))), dtype=tl.float32)
      r126_right = tl.load($(candidate_vto_copy_35_ptr) + (((((((r126_group % $(8)) * $(12288)) + (r126_lane[:, None] * $(64)) + r126_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r126_group % $(8)) * $(12288)) + (r126_lane[:, None] * $(64)) + r126_col[None, :]) % $(64)) * $(192)) + (((((r126_group % $(8)) * $(12288)) + (r126_lane[:, None] * $(64)) + r126_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r126_value = tl.dot(r126_left, r126_right)
      r126_value = r126_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r126_group * $(2048)) + (r126_row[:, None] * $(64)) + r126_col[None, :]), r126_value)
    }
    r127_z = tl.arange($(16384))
    r127_value = (tl.load($(candidate_vbmm_18_ptr) + ((((r127_z / $(2048)) % $(8)) * $(2048)) + (((r127_z / $(64)) % $(32)) * $(64)) + (r127_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(16384)])
    tl.store($(candidate_vmul_25_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(32))
    r128_value = (tl.cast(r128_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r128_value = tl.broadcast_to(r128_value, [$(32)])
    tl.store($(candidate_vadd_10_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(32))
    r129_value = (tl.load($(candidate_vadd_10_ptr) + r129_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r129_value = tl.broadcast_to(r129_value, [$(32)])
    tl.store($(candidate_vsub_19_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(2048))
    r130_value = (tl.cast((r130_z % $(64)), tl.int64) > tl.load($(candidate_vsub_19_ptr) + ((r130_z / $(64)) % $(32)), dtype=tl.int64))
    r130_value = tl.broadcast_to(r130_value, [$(2048)])
    tl.store($(candidate_vgt_4_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(16384))
    r131_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r131_z / $(64)) % $(32)) * $(64)) + (r131_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_25_ptr) + ((((r131_z / $(2048)) % $(8)) * $(2048)) + (((r131_z / $(64)) % $(32)) * $(64)) + (r131_z % $(64))), dtype=tl.float32))
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(16384))
    r132_lane = tl.arange($(64))
    r132_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r132_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r132_z[:, None] / $(64)) % $(32)) * $(64)) + (r132_lane[None, :] % $(64))), dtype=tl.float32)
    r132_maximum = tl.max(r132_row_values, axis=1, keep_dims=true)
    r132_weights = tl.exp(r132_row_values - r132_maximum)
    r132_total = tl.sum(r132_weights, axis=1)
    r132_maximum = tl.max(r132_row_values, axis=1)
    r132_value = tl.exp(tl.load($(candidate_vmasked_fill_4_ptr) + ((((r132_z / $(2048)) % $(8)) * $(2048)) + (((r132_z / $(64)) % $(32)) * $(64)) + (r132_z % $(64))), dtype=tl.float32) - r132_maximum) / r132_total
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(16384)])
    tl.store($(candidate_vsoftmax_4_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(16384))
    r133_value = tl.load($(candidate_vsoftmax_4_ptr) + ((((r133_z / $(2048)) % $(8)) * $(2048)) + (((r133_z / $(64)) % $(32)) * $(64)) + (r133_z % $(64))), dtype=tl.float32)
    r133_value = r133_value.to(tl.bfloat16)
    r133_value = tl.broadcast_to(r133_value, [$(16384)])
    tl.store($(candidate_vto_copy_36_ptr) + r133_z, r133_value)
    r134_row = tl.arange($(96))
    r134_col = tl.arange($(1024))
    r134_lane = tl.arange($(512))
    for r134_group in range($(1)) {
      r134_left = tl.load($(candidate_vcat_7_ptr) + ((((((r134_row[:, None] * $(512)) + r134_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r134_row[:, None] * $(512)) + r134_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r134_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r134_lane[:, None] * $(1024)) + r134_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r134_lane[:, None] * $(1024)) + r134_col[None, :]) % $(128)) * $(512)) + ((((r134_lane[:, None] * $(1024)) + r134_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r134_value = tl.dot(r134_left, r134_right)
      r134_value = r134_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_20_ptr) + ((r134_group * $(98304)) + (r134_row[:, None] * $(1024)) + r134_col[None, :]), r134_value)
    }
    r135_row = tl.arange($(96))
    r135_col = tl.arange($(1024))
    r135_lane = tl.arange($(512))
    for r135_group in range($(1)) {
      r135_left = tl.load($(candidate_vcat_7_ptr) + ((((((r135_row[:, None] * $(512)) + r135_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r135_row[:, None] * $(512)) + r135_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r135_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r135_lane[:, None] * $(1024)) + r135_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r135_lane[:, None] * $(1024)) + r135_col[None, :]) % $(128)) * $(512)) + ((((r135_lane[:, None] * $(1024)) + r135_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r135_value = tl.dot(r135_left, r135_right)
      r135_value = r135_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_21_ptr) + ((r135_group * $(98304)) + (r135_row[:, None] * $(1024)) + r135_col[None, :]), r135_value)
    }
    r136_z = tl.arange($(49152))
    r136_value = tl.where((r136_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r136_z / $(1536)) % $(32)) * $(1024)) + (((r136_z / $(192)) % $(8)) * $(128)) + ((r136_z % $(192)) - $(0)) + $(163840)) / $(1024)) % $(288)) * $(1536)) + (((((((r136_z / $(1536)) % $(32)) * $(1024)) + (((r136_z / $(192)) % $(8)) * $(128)) + ((r136_z % $(192)) - $(0)) + $(163840)) / $(128)) % $(8)) * $(192)) + (((((r136_z / $(1536)) % $(32)) * $(1024)) + (((r136_z / $(192)) % $(8)) * $(128)) + ((r136_z % $(192)) - $(0)) + $(163840)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r136_z / $(1536)) % $(32)) * $(512)) + (((r136_z / $(192)) % $(8)) * $(64)) + ((r136_z % $(192)) - $(128)) + $(81920)) / $(64)) % $(8)) * $(18432)) + (((((((r136_z / $(1536)) % $(32)) * $(512)) + (((r136_z / $(192)) % $(8)) * $(64)) + ((r136_z % $(192)) - $(128)) + $(81920)) / $(512)) % $(288)) * $(64)) + (((((r136_z / $(1536)) % $(32)) * $(512)) + (((r136_z / $(192)) % $(8)) * $(64)) + ((r136_z % $(192)) - $(128)) + $(81920)) % $(64))), dtype=tl.bfloat16))
    r136_value = r136_value.to(tl.bfloat16)
    r136_value = tl.broadcast_to(r136_value, [$(49152)])
    tl.store($(candidate_vcat_23_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(147456))
    r137_value = tl.where((r137_z % $(192)) < $(128), tl.load($(candidate_vbmm_20_ptr) + ((((((((r137_z / $(1536)) % $(96)) * $(1024)) + (((r137_z / $(192)) % $(8)) * $(128)) + ((r137_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r137_z / $(1536)) % $(96)) * $(1024)) + (((r137_z / $(192)) % $(8)) * $(128)) + ((r137_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r137_z / $(1536)) % $(96)) * $(1024)) + (((r137_z / $(192)) % $(8)) * $(128)) + ((r137_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r137_z / $(1536)) % $(96)) * $(576)) + ((((r137_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r137_value = r137_value.to(tl.bfloat16)
    r137_value = tl.broadcast_to(r137_value, [$(147456)])
    tl.store($(candidate_vcat_24_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(49152))
    r138_value = tl.load($(candidate_vcat_23_ptr) + ((((r138_z / $(192)) % $(32)) * $(1536)) + (((r138_z / $(6144)) % $(8)) * $(192)) + (r138_z % $(192))), dtype=tl.bfloat16)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(49152)])
    tl.store($(candidate_vto_copy_37_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(147456))
    r139_value = tl.load($(candidate_vcat_24_ptr) + ((((r139_z / $(192)) % $(96)) * $(1536)) + (((r139_z / $(18432)) % $(8)) * $(192)) + (r139_z % $(192))), dtype=tl.bfloat16)
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(147456)])
    tl.store($(candidate_vto_copy_38_ptr) + r139_z, r139_value)
    r140_row = tl.arange($(32))
    r140_col = tl.arange($(96))
    r140_lane = tl.arange($(192))
    for r140_group in range($(8)) {
      r140_left = tl.load($(candidate_vto_copy_37_ptr) + (((((((r140_group % $(8)) * $(6144)) + (r140_row[:, None] * $(192)) + r140_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r140_group % $(8)) * $(6144)) + (r140_row[:, None] * $(192)) + r140_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r140_group % $(8)) * $(6144)) + (r140_row[:, None] * $(192)) + r140_lane[None, :]) % $(192))), dtype=tl.float32)
      r140_right = tl.load($(candidate_vto_copy_38_ptr) + (((((((r140_group % $(8)) * $(18432)) + (r140_lane[:, None] * $(96)) + r140_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r140_group % $(8)) * $(18432)) + (r140_lane[:, None] * $(96)) + r140_col[None, :]) % $(96)) * $(192)) + (((((r140_group % $(8)) * $(18432)) + (r140_lane[:, None] * $(96)) + r140_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r140_value = tl.dot(r140_left, r140_right)
      r140_value = r140_value.to(tl.float32)
      tl.store($(candidate_vbmm_22_ptr) + ((r140_group * $(3072)) + (r140_row[:, None] * $(96)) + r140_col[None, :]), r140_value)
    }
    r141_z = tl.arange($(24576))
    r141_value = (tl.load($(candidate_vbmm_22_ptr) + ((((r141_z / $(3072)) % $(8)) * $(3072)) + (((r141_z / $(96)) % $(32)) * $(96)) + (r141_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(24576)])
    tl.store($(candidate_vmul_26_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(32))
    r142_value = (tl.cast(r142_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r142_value = tl.broadcast_to(r142_value, [$(32)])
    tl.store($(candidate_vadd_11_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(32))
    r143_value = (tl.load($(candidate_vadd_11_ptr) + r143_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r143_value = tl.broadcast_to(r143_value, [$(32)])
    tl.store($(candidate_vsub_20_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(3072))
    r144_value = (tl.cast((r144_z % $(96)), tl.int64) > tl.load($(candidate_vsub_20_ptr) + ((r144_z / $(96)) % $(32)), dtype=tl.int64))
    r144_value = tl.broadcast_to(r144_value, [$(3072)])
    tl.store($(candidate_vgt_5_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(24576))
    r145_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r145_z / $(96)) % $(32)) * $(96)) + (r145_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_26_ptr) + ((((r145_z / $(3072)) % $(8)) * $(3072)) + (((r145_z / $(96)) % $(32)) * $(96)) + (r145_z % $(96))), dtype=tl.float32))
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(24576))
    r146_lane = tl.arange($(96))
    r146_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r146_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r146_z[:, None] / $(96)) % $(32)) * $(96)) + (r146_lane[None, :] % $(96))), dtype=tl.float32)
    r146_maximum = tl.max(r146_row_values, axis=1, keep_dims=true)
    r146_weights = tl.exp(r146_row_values - r146_maximum)
    r146_total = tl.sum(r146_weights, axis=1)
    r146_maximum = tl.max(r146_row_values, axis=1)
    r146_value = tl.exp(tl.load($(candidate_vmasked_fill_5_ptr) + ((((r146_z / $(3072)) % $(8)) * $(3072)) + (((r146_z / $(96)) % $(32)) * $(96)) + (r146_z % $(96))), dtype=tl.float32) - r146_maximum) / r146_total
    r146_value = r146_value.to(tl.float32)
    r146_value = tl.broadcast_to(r146_value, [$(24576)])
    tl.store($(candidate_vsoftmax_5_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(24576))
    r147_value = tl.load($(candidate_vsoftmax_5_ptr) + ((((r147_z / $(3072)) % $(8)) * $(3072)) + (((r147_z / $(96)) % $(32)) * $(96)) + (r147_z % $(96))), dtype=tl.float32)
    r147_value = r147_value.to(tl.bfloat16)
    r147_value = tl.broadcast_to(r147_value, [$(24576)])
    tl.store($(candidate_vto_copy_39_ptr) + r147_z, r147_value)
    r148_row = tl.arange($(96))
    r148_col = tl.arange($(1024))
    r148_lane = tl.arange($(512))
    for r148_group in range($(1)) {
      r148_left = tl.load($(candidate_vcat_7_ptr) + ((((((r148_row[:, None] * $(512)) + r148_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r148_row[:, None] * $(512)) + r148_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r148_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r148_lane[:, None] * $(1024)) + r148_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r148_lane[:, None] * $(1024)) + r148_col[None, :]) % $(128)) * $(512)) + ((((r148_lane[:, None] * $(1024)) + r148_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r148_value = tl.dot(r148_left, r148_right)
      r148_value = r148_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_24_ptr) + ((r148_group * $(98304)) + (r148_row[:, None] * $(1024)) + r148_col[None, :]), r148_value)
    }
    r149_row = tl.arange($(96))
    r149_col = tl.arange($(1024))
    r149_lane = tl.arange($(512))
    for r149_group in range($(1)) {
      r149_left = tl.load($(candidate_vcat_7_ptr) + ((((((r149_row[:, None] * $(512)) + r149_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r149_row[:, None] * $(512)) + r149_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r149_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r149_lane[:, None] * $(1024)) + r149_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r149_lane[:, None] * $(1024)) + r149_col[None, :]) % $(128)) * $(512)) + ((((r149_lane[:, None] * $(1024)) + r149_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r149_value = tl.dot(r149_left, r149_right)
      r149_value = r149_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_25_ptr) + ((r149_group * $(98304)) + (r149_row[:, None] * $(1024)) + r149_col[None, :]), r149_value)
    }
    r150_z = tl.arange($(49152))
    r150_value = tl.where((r150_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r150_z / $(1536)) % $(32)) * $(1024)) + (((r150_z / $(192)) % $(8)) * $(128)) + ((r150_z % $(192)) - $(0)) + $(196608)) / $(1024)) % $(288)) * $(1536)) + (((((((r150_z / $(1536)) % $(32)) * $(1024)) + (((r150_z / $(192)) % $(8)) * $(128)) + ((r150_z % $(192)) - $(0)) + $(196608)) / $(128)) % $(8)) * $(192)) + (((((r150_z / $(1536)) % $(32)) * $(1024)) + (((r150_z / $(192)) % $(8)) * $(128)) + ((r150_z % $(192)) - $(0)) + $(196608)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r150_z / $(1536)) % $(32)) * $(512)) + (((r150_z / $(192)) % $(8)) * $(64)) + ((r150_z % $(192)) - $(128)) + $(98304)) / $(64)) % $(8)) * $(18432)) + (((((((r150_z / $(1536)) % $(32)) * $(512)) + (((r150_z / $(192)) % $(8)) * $(64)) + ((r150_z % $(192)) - $(128)) + $(98304)) / $(512)) % $(288)) * $(64)) + (((((r150_z / $(1536)) % $(32)) * $(512)) + (((r150_z / $(192)) % $(8)) * $(64)) + ((r150_z % $(192)) - $(128)) + $(98304)) % $(64))), dtype=tl.bfloat16))
    r150_value = r150_value.to(tl.bfloat16)
    r150_value = tl.broadcast_to(r150_value, [$(49152)])
    tl.store($(candidate_vcat_26_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(147456))
    r151_value = tl.where((r151_z % $(192)) < $(128), tl.load($(candidate_vbmm_24_ptr) + ((((((((r151_z / $(1536)) % $(96)) * $(1024)) + (((r151_z / $(192)) % $(8)) * $(128)) + ((r151_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r151_z / $(1536)) % $(96)) * $(1024)) + (((r151_z / $(192)) % $(8)) * $(128)) + ((r151_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r151_z / $(1536)) % $(96)) * $(1024)) + (((r151_z / $(192)) % $(8)) * $(128)) + ((r151_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r151_z / $(1536)) % $(96)) * $(576)) + ((((r151_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r151_value = r151_value.to(tl.bfloat16)
    r151_value = tl.broadcast_to(r151_value, [$(147456)])
    tl.store($(candidate_vcat_27_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(49152))
    r152_value = tl.load($(candidate_vcat_26_ptr) + ((((r152_z / $(192)) % $(32)) * $(1536)) + (((r152_z / $(6144)) % $(8)) * $(192)) + (r152_z % $(192))), dtype=tl.bfloat16)
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(49152)])
    tl.store($(candidate_vto_copy_40_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(49152))
    r153_value = tl.load($(candidate_vcat_27_ptr) + ((((r153_z / $(192)) % $(32)) * $(1536)) + (((r153_z / $(6144)) % $(8)) * $(192)) + (r153_z % $(192))), dtype=tl.bfloat16)
    r153_value = r153_value.to(tl.float32)
    r153_value = tl.broadcast_to(r153_value, [$(49152)])
    tl.store($(candidate_vto_copy_41_ptr) + r153_z, r153_value)
    r154_row = tl.arange($(32))
    r154_col = tl.arange($(32))
    r154_lane = tl.arange($(192))
    for r154_group in range($(8)) {
      r154_left = tl.load($(candidate_vto_copy_40_ptr) + (((((((r154_group % $(8)) * $(6144)) + (r154_row[:, None] * $(192)) + r154_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r154_group % $(8)) * $(6144)) + (r154_row[:, None] * $(192)) + r154_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r154_group % $(8)) * $(6144)) + (r154_row[:, None] * $(192)) + r154_lane[None, :]) % $(192))), dtype=tl.float32)
      r154_right = tl.load($(candidate_vto_copy_41_ptr) + (((((((r154_group % $(8)) * $(6144)) + (r154_lane[:, None] * $(32)) + r154_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r154_group % $(8)) * $(6144)) + (r154_lane[:, None] * $(32)) + r154_col[None, :]) % $(32)) * $(192)) + (((((r154_group % $(8)) * $(6144)) + (r154_lane[:, None] * $(32)) + r154_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r154_value = tl.dot(r154_left, r154_right)
      r154_value = r154_value.to(tl.float32)
      tl.store($(candidate_vbmm_26_ptr) + ((r154_group * $(1024)) + (r154_row[:, None] * $(32)) + r154_col[None, :]), r154_value)
    }
    r155_z = tl.arange($(8192))
    r155_value = (tl.load($(candidate_vbmm_26_ptr) + ((((r155_z / $(1024)) % $(8)) * $(1024)) + (((r155_z / $(32)) % $(32)) * $(32)) + (r155_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(8192)])
    tl.store($(candidate_vmul_27_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(32))
    r156_value = (tl.cast(r156_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r156_value = tl.broadcast_to(r156_value, [$(32)])
    tl.store($(candidate_vadd_12_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(32))
    r157_value = (tl.load($(candidate_vadd_12_ptr) + r157_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r157_value = tl.broadcast_to(r157_value, [$(32)])
    tl.store($(candidate_vsub_21_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(1024))
    r158_value = (tl.cast((r158_z % $(32)), tl.int64) > tl.load($(candidate_vsub_21_ptr) + ((r158_z / $(32)) % $(32)), dtype=tl.int64))
    r158_value = tl.broadcast_to(r158_value, [$(1024)])
    tl.store($(candidate_vgt_6_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(8192))
    r159_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r159_z / $(32)) % $(32)) * $(32)) + (r159_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_27_ptr) + ((((r159_z / $(1024)) % $(8)) * $(1024)) + (((r159_z / $(32)) % $(32)) * $(32)) + (r159_z % $(32))), dtype=tl.float32))
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_6_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(8192))
    r160_lane = tl.arange($(32))
    r160_row_values = tl.load($(candidate_vmasked_fill_6_ptr) + ((((r160_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r160_z[:, None] / $(32)) % $(32)) * $(32)) + (r160_lane[None, :] % $(32))), dtype=tl.float32)
    r160_maximum = tl.max(r160_row_values, axis=1, keep_dims=true)
    r160_weights = tl.exp(r160_row_values - r160_maximum)
    r160_total = tl.sum(r160_weights, axis=1)
    r160_maximum = tl.max(r160_row_values, axis=1)
    r160_value = tl.exp(tl.load($(candidate_vmasked_fill_6_ptr) + ((((r160_z / $(1024)) % $(8)) * $(1024)) + (((r160_z / $(32)) % $(32)) * $(32)) + (r160_z % $(32))), dtype=tl.float32) - r160_maximum) / r160_total
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(8192)])
    tl.store($(candidate_vsoftmax_6_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(8192))
    r161_value = tl.load($(candidate_vsoftmax_6_ptr) + ((((r161_z / $(1024)) % $(8)) * $(1024)) + (((r161_z / $(32)) % $(32)) * $(32)) + (r161_z % $(32))), dtype=tl.float32)
    r161_value = r161_value.to(tl.bfloat16)
    r161_value = tl.broadcast_to(r161_value, [$(8192)])
    tl.store($(candidate_vto_copy_42_ptr) + r161_z, r161_value)
    r162_row = tl.arange($(96))
    r162_col = tl.arange($(1024))
    r162_lane = tl.arange($(512))
    for r162_group in range($(1)) {
      r162_left = tl.load($(candidate_vcat_7_ptr) + ((((((r162_row[:, None] * $(512)) + r162_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r162_row[:, None] * $(512)) + r162_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r162_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r162_lane[:, None] * $(1024)) + r162_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r162_lane[:, None] * $(1024)) + r162_col[None, :]) % $(128)) * $(512)) + ((((r162_lane[:, None] * $(1024)) + r162_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r162_value = tl.dot(r162_left, r162_right)
      r162_value = r162_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_28_ptr) + ((r162_group * $(98304)) + (r162_row[:, None] * $(1024)) + r162_col[None, :]), r162_value)
    }
    r163_row = tl.arange($(96))
    r163_col = tl.arange($(1024))
    r163_lane = tl.arange($(512))
    for r163_group in range($(1)) {
      r163_left = tl.load($(candidate_vcat_7_ptr) + ((((((r163_row[:, None] * $(512)) + r163_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r163_row[:, None] * $(512)) + r163_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r163_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r163_lane[:, None] * $(1024)) + r163_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r163_lane[:, None] * $(1024)) + r163_col[None, :]) % $(128)) * $(512)) + ((((r163_lane[:, None] * $(1024)) + r163_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r163_value = tl.dot(r163_left, r163_right)
      r163_value = r163_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_29_ptr) + ((r163_group * $(98304)) + (r163_row[:, None] * $(1024)) + r163_col[None, :]), r163_value)
    }
    r164_z = tl.arange($(49152))
    r164_value = tl.where((r164_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r164_z / $(1536)) % $(32)) * $(1024)) + (((r164_z / $(192)) % $(8)) * $(128)) + ((r164_z % $(192)) - $(0)) + $(229376)) / $(1024)) % $(288)) * $(1536)) + (((((((r164_z / $(1536)) % $(32)) * $(1024)) + (((r164_z / $(192)) % $(8)) * $(128)) + ((r164_z % $(192)) - $(0)) + $(229376)) / $(128)) % $(8)) * $(192)) + (((((r164_z / $(1536)) % $(32)) * $(1024)) + (((r164_z / $(192)) % $(8)) * $(128)) + ((r164_z % $(192)) - $(0)) + $(229376)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r164_z / $(1536)) % $(32)) * $(512)) + (((r164_z / $(192)) % $(8)) * $(64)) + ((r164_z % $(192)) - $(128)) + $(114688)) / $(64)) % $(8)) * $(18432)) + (((((((r164_z / $(1536)) % $(32)) * $(512)) + (((r164_z / $(192)) % $(8)) * $(64)) + ((r164_z % $(192)) - $(128)) + $(114688)) / $(512)) % $(288)) * $(64)) + (((((r164_z / $(1536)) % $(32)) * $(512)) + (((r164_z / $(192)) % $(8)) * $(64)) + ((r164_z % $(192)) - $(128)) + $(114688)) % $(64))), dtype=tl.bfloat16))
    r164_value = r164_value.to(tl.bfloat16)
    r164_value = tl.broadcast_to(r164_value, [$(49152)])
    tl.store($(candidate_vcat_29_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(147456))
    r165_value = tl.where((r165_z % $(192)) < $(128), tl.load($(candidate_vbmm_28_ptr) + ((((((((r165_z / $(1536)) % $(96)) * $(1024)) + (((r165_z / $(192)) % $(8)) * $(128)) + ((r165_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r165_z / $(1536)) % $(96)) * $(1024)) + (((r165_z / $(192)) % $(8)) * $(128)) + ((r165_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r165_z / $(1536)) % $(96)) * $(1024)) + (((r165_z / $(192)) % $(8)) * $(128)) + ((r165_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r165_z / $(1536)) % $(96)) * $(576)) + ((((r165_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r165_value = r165_value.to(tl.bfloat16)
    r165_value = tl.broadcast_to(r165_value, [$(147456)])
    tl.store($(candidate_vcat_30_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(49152))
    r166_value = tl.load($(candidate_vcat_29_ptr) + ((((r166_z / $(192)) % $(32)) * $(1536)) + (((r166_z / $(6144)) % $(8)) * $(192)) + (r166_z % $(192))), dtype=tl.bfloat16)
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(49152)])
    tl.store($(candidate_vto_copy_43_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(98304))
    r167_value = tl.load($(candidate_vcat_30_ptr) + ((((r167_z / $(192)) % $(64)) * $(1536)) + (((r167_z / $(12288)) % $(8)) * $(192)) + (r167_z % $(192))), dtype=tl.bfloat16)
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(98304)])
    tl.store($(candidate_vto_copy_44_ptr) + r167_z, r167_value)
    r168_row = tl.arange($(32))
    r168_col = tl.arange($(64))
    r168_lane = tl.arange($(192))
    for r168_group in range($(8)) {
      r168_left = tl.load($(candidate_vto_copy_43_ptr) + (((((((r168_group % $(8)) * $(6144)) + (r168_row[:, None] * $(192)) + r168_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r168_group % $(8)) * $(6144)) + (r168_row[:, None] * $(192)) + r168_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r168_group % $(8)) * $(6144)) + (r168_row[:, None] * $(192)) + r168_lane[None, :]) % $(192))), dtype=tl.float32)
      r168_right = tl.load($(candidate_vto_copy_44_ptr) + (((((((r168_group % $(8)) * $(12288)) + (r168_lane[:, None] * $(64)) + r168_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r168_group % $(8)) * $(12288)) + (r168_lane[:, None] * $(64)) + r168_col[None, :]) % $(64)) * $(192)) + (((((r168_group % $(8)) * $(12288)) + (r168_lane[:, None] * $(64)) + r168_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r168_value = tl.dot(r168_left, r168_right)
      r168_value = r168_value.to(tl.float32)
      tl.store($(candidate_vbmm_30_ptr) + ((r168_group * $(2048)) + (r168_row[:, None] * $(64)) + r168_col[None, :]), r168_value)
    }
    r169_z = tl.arange($(16384))
    r169_value = (tl.load($(candidate_vbmm_30_ptr) + ((((r169_z / $(2048)) % $(8)) * $(2048)) + (((r169_z / $(64)) % $(32)) * $(64)) + (r169_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(16384)])
    tl.store($(candidate_vmul_28_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(32))
    r170_value = (tl.cast(r170_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r170_value = tl.broadcast_to(r170_value, [$(32)])
    tl.store($(candidate_vadd_13_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(32))
    r171_value = (tl.load($(candidate_vadd_13_ptr) + r171_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r171_value = tl.broadcast_to(r171_value, [$(32)])
    tl.store($(candidate_vsub_22_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(2048))
    r172_value = (tl.cast((r172_z % $(64)), tl.int64) > tl.load($(candidate_vsub_22_ptr) + ((r172_z / $(64)) % $(32)), dtype=tl.int64))
    r172_value = tl.broadcast_to(r172_value, [$(2048)])
    tl.store($(candidate_vgt_7_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(16384))
    r173_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r173_z / $(64)) % $(32)) * $(64)) + (r173_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_28_ptr) + ((((r173_z / $(2048)) % $(8)) * $(2048)) + (((r173_z / $(64)) % $(32)) * $(64)) + (r173_z % $(64))), dtype=tl.float32))
    r173_value = r173_value.to(tl.float32)
    r173_value = tl.broadcast_to(r173_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_7_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(16384))
    r174_lane = tl.arange($(64))
    r174_row_values = tl.load($(candidate_vmasked_fill_7_ptr) + ((((r174_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r174_z[:, None] / $(64)) % $(32)) * $(64)) + (r174_lane[None, :] % $(64))), dtype=tl.float32)
    r174_maximum = tl.max(r174_row_values, axis=1, keep_dims=true)
    r174_weights = tl.exp(r174_row_values - r174_maximum)
    r174_total = tl.sum(r174_weights, axis=1)
    r174_maximum = tl.max(r174_row_values, axis=1)
    r174_value = tl.exp(tl.load($(candidate_vmasked_fill_7_ptr) + ((((r174_z / $(2048)) % $(8)) * $(2048)) + (((r174_z / $(64)) % $(32)) * $(64)) + (r174_z % $(64))), dtype=tl.float32) - r174_maximum) / r174_total
    r174_value = r174_value.to(tl.float32)
    r174_value = tl.broadcast_to(r174_value, [$(16384)])
    tl.store($(candidate_vsoftmax_7_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(16384))
    r175_value = tl.load($(candidate_vsoftmax_7_ptr) + ((((r175_z / $(2048)) % $(8)) * $(2048)) + (((r175_z / $(64)) % $(32)) * $(64)) + (r175_z % $(64))), dtype=tl.float32)
    r175_value = r175_value.to(tl.bfloat16)
    r175_value = tl.broadcast_to(r175_value, [$(16384)])
    tl.store($(candidate_vto_copy_45_ptr) + r175_z, r175_value)
    r176_row = tl.arange($(96))
    r176_col = tl.arange($(1024))
    r176_lane = tl.arange($(512))
    for r176_group in range($(1)) {
      r176_left = tl.load($(candidate_vcat_7_ptr) + ((((((r176_row[:, None] * $(512)) + r176_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r176_row[:, None] * $(512)) + r176_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r176_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r176_lane[:, None] * $(1024)) + r176_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r176_lane[:, None] * $(1024)) + r176_col[None, :]) % $(128)) * $(512)) + ((((r176_lane[:, None] * $(1024)) + r176_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r176_value = tl.dot(r176_left, r176_right)
      r176_value = r176_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_32_ptr) + ((r176_group * $(98304)) + (r176_row[:, None] * $(1024)) + r176_col[None, :]), r176_value)
    }
    r177_row = tl.arange($(96))
    r177_col = tl.arange($(1024))
    r177_lane = tl.arange($(512))
    for r177_group in range($(1)) {
      r177_left = tl.load($(candidate_vcat_7_ptr) + ((((((r177_row[:, None] * $(512)) + r177_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r177_row[:, None] * $(512)) + r177_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r177_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r177_lane[:, None] * $(1024)) + r177_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r177_lane[:, None] * $(1024)) + r177_col[None, :]) % $(128)) * $(512)) + ((((r177_lane[:, None] * $(1024)) + r177_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r177_value = tl.dot(r177_left, r177_right)
      r177_value = r177_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_33_ptr) + ((r177_group * $(98304)) + (r177_row[:, None] * $(1024)) + r177_col[None, :]), r177_value)
    }
    r178_z = tl.arange($(49152))
    r178_value = tl.where((r178_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r178_z / $(1536)) % $(32)) * $(1024)) + (((r178_z / $(192)) % $(8)) * $(128)) + ((r178_z % $(192)) - $(0)) + $(262144)) / $(1024)) % $(288)) * $(1536)) + (((((((r178_z / $(1536)) % $(32)) * $(1024)) + (((r178_z / $(192)) % $(8)) * $(128)) + ((r178_z % $(192)) - $(0)) + $(262144)) / $(128)) % $(8)) * $(192)) + (((((r178_z / $(1536)) % $(32)) * $(1024)) + (((r178_z / $(192)) % $(8)) * $(128)) + ((r178_z % $(192)) - $(0)) + $(262144)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r178_z / $(1536)) % $(32)) * $(512)) + (((r178_z / $(192)) % $(8)) * $(64)) + ((r178_z % $(192)) - $(128)) + $(131072)) / $(64)) % $(8)) * $(18432)) + (((((((r178_z / $(1536)) % $(32)) * $(512)) + (((r178_z / $(192)) % $(8)) * $(64)) + ((r178_z % $(192)) - $(128)) + $(131072)) / $(512)) % $(288)) * $(64)) + (((((r178_z / $(1536)) % $(32)) * $(512)) + (((r178_z / $(192)) % $(8)) * $(64)) + ((r178_z % $(192)) - $(128)) + $(131072)) % $(64))), dtype=tl.bfloat16))
    r178_value = r178_value.to(tl.bfloat16)
    r178_value = tl.broadcast_to(r178_value, [$(49152)])
    tl.store($(candidate_vcat_32_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(147456))
    r179_value = tl.where((r179_z % $(192)) < $(128), tl.load($(candidate_vbmm_32_ptr) + ((((((((r179_z / $(1536)) % $(96)) * $(1024)) + (((r179_z / $(192)) % $(8)) * $(128)) + ((r179_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r179_z / $(1536)) % $(96)) * $(1024)) + (((r179_z / $(192)) % $(8)) * $(128)) + ((r179_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r179_z / $(1536)) % $(96)) * $(1024)) + (((r179_z / $(192)) % $(8)) * $(128)) + ((r179_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r179_z / $(1536)) % $(96)) * $(576)) + ((((r179_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r179_value = r179_value.to(tl.bfloat16)
    r179_value = tl.broadcast_to(r179_value, [$(147456)])
    tl.store($(candidate_vcat_33_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(49152))
    r180_value = tl.load($(candidate_vcat_32_ptr) + ((((r180_z / $(192)) % $(32)) * $(1536)) + (((r180_z / $(6144)) % $(8)) * $(192)) + (r180_z % $(192))), dtype=tl.bfloat16)
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(49152)])
    tl.store($(candidate_vto_copy_46_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(147456))
    r181_value = tl.load($(candidate_vcat_33_ptr) + ((((r181_z / $(192)) % $(96)) * $(1536)) + (((r181_z / $(18432)) % $(8)) * $(192)) + (r181_z % $(192))), dtype=tl.bfloat16)
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(147456)])
    tl.store($(candidate_vto_copy_47_ptr) + r181_z, r181_value)
    r182_row = tl.arange($(32))
    r182_col = tl.arange($(96))
    r182_lane = tl.arange($(192))
    for r182_group in range($(8)) {
      r182_left = tl.load($(candidate_vto_copy_46_ptr) + (((((((r182_group % $(8)) * $(6144)) + (r182_row[:, None] * $(192)) + r182_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r182_group % $(8)) * $(6144)) + (r182_row[:, None] * $(192)) + r182_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r182_group % $(8)) * $(6144)) + (r182_row[:, None] * $(192)) + r182_lane[None, :]) % $(192))), dtype=tl.float32)
      r182_right = tl.load($(candidate_vto_copy_47_ptr) + (((((((r182_group % $(8)) * $(18432)) + (r182_lane[:, None] * $(96)) + r182_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r182_group % $(8)) * $(18432)) + (r182_lane[:, None] * $(96)) + r182_col[None, :]) % $(96)) * $(192)) + (((((r182_group % $(8)) * $(18432)) + (r182_lane[:, None] * $(96)) + r182_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r182_value = tl.dot(r182_left, r182_right)
      r182_value = r182_value.to(tl.float32)
      tl.store($(candidate_vbmm_34_ptr) + ((r182_group * $(3072)) + (r182_row[:, None] * $(96)) + r182_col[None, :]), r182_value)
    }
    r183_z = tl.arange($(24576))
    r183_value = (tl.load($(candidate_vbmm_34_ptr) + ((((r183_z / $(3072)) % $(8)) * $(3072)) + (((r183_z / $(96)) % $(32)) * $(96)) + (r183_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(24576)])
    tl.store($(candidate_vmul_29_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(32))
    r184_value = (tl.cast(r184_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r184_value = tl.broadcast_to(r184_value, [$(32)])
    tl.store($(candidate_vadd_14_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(32))
    r185_value = (tl.load($(candidate_vadd_14_ptr) + r185_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r185_value = tl.broadcast_to(r185_value, [$(32)])
    tl.store($(candidate_vsub_23_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(3072))
    r186_value = (tl.cast((r186_z % $(96)), tl.int64) > tl.load($(candidate_vsub_23_ptr) + ((r186_z / $(96)) % $(32)), dtype=tl.int64))
    r186_value = tl.broadcast_to(r186_value, [$(3072)])
    tl.store($(candidate_vgt_8_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(24576))
    r187_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r187_z / $(96)) % $(32)) * $(96)) + (r187_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_29_ptr) + ((((r187_z / $(3072)) % $(8)) * $(3072)) + (((r187_z / $(96)) % $(32)) * $(96)) + (r187_z % $(96))), dtype=tl.float32))
    r187_value = r187_value.to(tl.float32)
    r187_value = tl.broadcast_to(r187_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_8_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(24576))
    r188_lane = tl.arange($(96))
    r188_row_values = tl.load($(candidate_vmasked_fill_8_ptr) + ((((r188_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r188_z[:, None] / $(96)) % $(32)) * $(96)) + (r188_lane[None, :] % $(96))), dtype=tl.float32)
    r188_maximum = tl.max(r188_row_values, axis=1, keep_dims=true)
    r188_weights = tl.exp(r188_row_values - r188_maximum)
    r188_total = tl.sum(r188_weights, axis=1)
    r188_maximum = tl.max(r188_row_values, axis=1)
    r188_value = tl.exp(tl.load($(candidate_vmasked_fill_8_ptr) + ((((r188_z / $(3072)) % $(8)) * $(3072)) + (((r188_z / $(96)) % $(32)) * $(96)) + (r188_z % $(96))), dtype=tl.float32) - r188_maximum) / r188_total
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(24576)])
    tl.store($(candidate_vsoftmax_8_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(24576))
    r189_value = tl.load($(candidate_vsoftmax_8_ptr) + ((((r189_z / $(3072)) % $(8)) * $(3072)) + (((r189_z / $(96)) % $(32)) * $(96)) + (r189_z % $(96))), dtype=tl.float32)
    r189_value = r189_value.to(tl.bfloat16)
    r189_value = tl.broadcast_to(r189_value, [$(24576)])
    tl.store($(candidate_vto_copy_48_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(147456))
    r190_value = tl.cast(0.0, tl.bfloat16)
    for r190_lane in range($(96)) {
      r190_value = tl.where(tl.load($(candidate_vcat_4_ptr) + (r190_lane % $(96)), dtype=tl.int64) == ((r190_z / $(512)) % $(288)), r190_value + tl.load($(input1_ptr) + (((r190_lane % $(96)) * $(512)) + (r190_z % $(512))), dtype=tl.bfloat16), r190_value)
    }
    r190_value = r190_value.to(tl.bfloat16)
    r190_value = tl.broadcast_to(r190_value, [$(147456)])
    tl.store($(candidate_vindex_put_ptr) + r190_z, r190_value)
    r191_row = tl.arange($(288))
    r191_col = tl.arange($(1024))
    r191_lane = tl.arange($(512))
    r191_left = tl.load($(candidate_vindex_put_ptr) + ((((((r191_row[:, None] * $(512)) + r191_lane[None, :]) / $(512)) % $(288)) * $(512)) + (((r191_row[:, None] * $(512)) + r191_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r191_right = tl.load($(candidate_vto_copy_6_ptr) + (((r191_lane[:, None] % $(512)) * $(1024)) + (r191_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r191_value = tl.dot(r191_left, r191_right)
    r191_value = r191_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r191_row[:, None] * $(1024)) + r191_col[None, :]), r191_value)
    r192_row = tl.arange($(96))
    r192_col = tl.arange($(128))
    r192_lane = tl.arange($(32))
    for r192_group in range($(8)) {
      r192_left = tl.load($(candidate_vto_copy_48_ptr) + (((((((r192_group % $(8)) * $(3072)) + (r192_lane[None, :] * $(96)) + r192_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r192_group % $(8)) * $(3072)) + (r192_lane[None, :] * $(96)) + r192_row[:, None]) / $(96)) % $(32)) * $(96)) + ((((r192_group % $(8)) * $(3072)) + (r192_lane[None, :] * $(96)) + r192_row[:, None]) % $(96))), dtype=tl.bfloat16)
      r192_right = tl.load($(candidate_vmm_4_ptr) + (((((((r192_group % $(8)) * $(4096)) + (r192_lane[:, None] * $(128)) + r192_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r192_group % $(8)) * $(4096)) + (r192_lane[:, None] * $(128)) + r192_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r192_group % $(8)) * $(4096)) + (r192_lane[:, None] * $(128)) + r192_col[None, :]) % $(128)) + $(262144)), dtype=tl.bfloat16)
      r192_value = tl.dot(r192_left, r192_right)
      r192_value = r192_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_36_ptr) + ((r192_group * $(12288)) + (r192_row[:, None] * $(128)) + r192_col[None, :]), r192_value)
    }
    r193_row = tl.arange($(32))
    r193_col = tl.arange($(96))
    r193_lane = tl.arange($(128))
    for r193_group in range($(8)) {
      r193_left = tl.load($(candidate_vmm_4_ptr) + (((((((r193_group % $(8)) * $(4096)) + (r193_row[:, None] * $(128)) + r193_lane[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r193_group % $(8)) * $(4096)) + (r193_row[:, None] * $(128)) + r193_lane[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r193_group % $(8)) * $(4096)) + (r193_row[:, None] * $(128)) + r193_lane[None, :]) % $(128)) + $(262144)), dtype=tl.bfloat16)
      r193_right = tl.load($(candidate_vbmm_33_ptr) + (((((((r193_group % $(8)) * $(12288)) + (r193_col[None, :] * $(128)) + r193_lane[:, None]) / $(128)) % $(96)) * $(1024)) + ((((((r193_group % $(8)) * $(12288)) + (r193_col[None, :] * $(128)) + r193_lane[:, None]) / $(12288)) % $(8)) * $(128)) + ((((r193_group % $(8)) * $(12288)) + (r193_col[None, :] * $(128)) + r193_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r193_value = tl.dot(r193_left, r193_right)
      r193_value = r193_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_37_ptr) + ((r193_group * $(3072)) + (r193_row[:, None] * $(96)) + r193_col[None, :]), r193_value)
    }
    r194_z = tl.arange($(24576))
    r194_value = tl.load($(candidate_vbmm_37_ptr) + ((((r194_z / $(3072)) % $(8)) * $(3072)) + (((r194_z / $(96)) % $(32)) * $(96)) + (r194_z % $(96))), dtype=tl.bfloat16)
    r194_value = r194_value.to(tl.float32)
    r194_value = tl.broadcast_to(r194_value, [$(24576)])
    tl.store($(candidate_vto_copy_49_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(24576))
    r195_lane = tl.arange($(96))
    r195_row_values = tl.load($(candidate_vsoftmax_8_ptr) + ((((r195_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r195_z[:, None] / $(96)) % $(32)) * $(96)) + (r195_lane[None, :] % $(96))), dtype=tl.float32)
    r195_row_grad = tl.load($(candidate_vto_copy_49_ptr) + ((((r195_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r195_z[:, None] / $(96)) % $(32)) * $(96)) + (r195_lane[None, :] % $(96))), dtype=tl.float32)
    r195_weighted = r195_row_grad * r195_row_values
    r195_dot_grad = tl.sum(r195_weighted, axis=1)
    r195_value = (tl.load($(candidate_vto_copy_49_ptr) + ((((r195_z / $(3072)) % $(8)) * $(3072)) + (((r195_z / $(96)) % $(32)) * $(96)) + (r195_z % $(96))), dtype=tl.float32) - r195_dot_grad) * tl.load($(candidate_vsoftmax_8_ptr) + ((((r195_z / $(3072)) % $(8)) * $(3072)) + (((r195_z / $(96)) % $(32)) * $(96)) + (r195_z % $(96))), dtype=tl.float32)
    r195_value = r195_value.to(tl.float32)
    r195_value = tl.broadcast_to(r195_value, [$(24576)])
    tl.store($(candidate_vsoftmax_backward_data_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(24576))
    r196_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r196_z / $(96)) % $(32)) * $(96)) + (r196_z % $(96))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_ptr) + ((((r196_z / $(3072)) % $(8)) * $(3072)) + (((r196_z / $(96)) % $(32)) * $(96)) + (r196_z % $(96))), dtype=tl.float32))
    r196_value = r196_value.to(tl.float32)
    r196_value = tl.broadcast_to(r196_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_9_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(24576))
    r197_value = (tl.load($(candidate_vmasked_fill_9_ptr) + ((((r197_z / $(3072)) % $(8)) * $(3072)) + (((r197_z / $(96)) % $(32)) * $(96)) + (r197_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r197_value = r197_value.to(tl.float32)
    r197_value = tl.broadcast_to(r197_value, [$(24576)])
    tl.store($(candidate_vmul_30_ptr) + r197_z, r197_value)
    r198_row = tl.arange($(192))
    r198_col = tl.arange($(96))
    r198_lane = tl.arange($(32))
    for r198_group in range($(8)) {
      r198_left = tl.load($(candidate_vto_copy_46_ptr) + (((((((r198_group % $(8)) * $(6144)) + (r198_lane[None, :] * $(192)) + r198_row[:, None]) / $(6144)) % $(8)) * $(6144)) + ((((((r198_group % $(8)) * $(6144)) + (r198_lane[None, :] * $(192)) + r198_row[:, None]) / $(192)) % $(32)) * $(192)) + ((((r198_group % $(8)) * $(6144)) + (r198_lane[None, :] * $(192)) + r198_row[:, None]) % $(192))), dtype=tl.float32)
      r198_right = tl.load($(candidate_vmul_30_ptr) + (((((((r198_group % $(8)) * $(3072)) + (r198_lane[:, None] * $(96)) + r198_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r198_group % $(8)) * $(3072)) + (r198_lane[:, None] * $(96)) + r198_col[None, :]) / $(96)) % $(32)) * $(96)) + ((((r198_group % $(8)) * $(3072)) + (r198_lane[:, None] * $(96)) + r198_col[None, :]) % $(96))), dtype=tl.float32)
      r198_value = tl.dot(r198_left, r198_right)
      r198_value = r198_value.to(tl.float32)
      tl.store($(candidate_vbmm_38_ptr) + ((r198_group * $(18432)) + (r198_row[:, None] * $(96)) + r198_col[None, :]), r198_value)
    }
    r199_row = tl.arange($(32))
    r199_col = tl.arange($(192))
    r199_lane = tl.arange($(96))
    for r199_group in range($(8)) {
      r199_left = tl.load($(candidate_vmul_30_ptr) + (((((((r199_group % $(8)) * $(3072)) + (r199_row[:, None] * $(96)) + r199_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r199_group % $(8)) * $(3072)) + (r199_row[:, None] * $(96)) + r199_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r199_group % $(8)) * $(3072)) + (r199_row[:, None] * $(96)) + r199_lane[None, :]) % $(96))), dtype=tl.float32)
      r199_right = tl.load($(candidate_vto_copy_47_ptr) + (((((((r199_group % $(8)) * $(18432)) + (r199_col[None, :] * $(96)) + r199_lane[:, None]) / $(18432)) % $(8)) * $(18432)) + (((((r199_group % $(8)) * $(18432)) + (r199_col[None, :] * $(96)) + r199_lane[:, None]) % $(96)) * $(192)) + (((((r199_group % $(8)) * $(18432)) + (r199_col[None, :] * $(96)) + r199_lane[:, None]) / $(96)) % $(192))), dtype=tl.float32)
      r199_value = tl.dot(r199_left, r199_right)
      r199_value = r199_value.to(tl.float32)
      tl.store($(candidate_vbmm_39_ptr) + ((r199_group * $(6144)) + (r199_row[:, None] * $(192)) + r199_col[None, :]), r199_value)
    }
    r200_z = tl.arange($(98304))
    r200_value = tl.where(((((r200_z / $(1024)) % $(96)) >= $(0)) & (((r200_z / $(1024)) % $(96)) < $(96)) & ((((r200_z / $(1024)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_36_ptr) + ((((((((r200_z / $(128)) % $(8)) * $(12288)) + ((((r200_z / $(1024)) % $(96)) - $(0)) * $(128)) + (r200_z % $(128))) / $(12288)) % $(8)) * $(12288)) + ((((((r200_z / $(128)) % $(8)) * $(96)) + (((r200_z / $(1024)) % $(96)) - $(0))) % $(96)) * $(128)) + (r200_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r200_value = r200_value.to(tl.bfloat16)
    r200_value = tl.broadcast_to(r200_value, [$(98304)])
    tl.store($(candidate_vslice_backward_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(147456))
    r201_value = tl.load($(candidate_vbmm_38_ptr) + ((((r201_z / $(18432)) % $(8)) * $(18432)) + ((r201_z % $(192)) * $(96)) + ((r201_z / $(192)) % $(96))), dtype=tl.float32)
    r201_value = r201_value.to(tl.bfloat16)
    r201_value = tl.broadcast_to(r201_value, [$(147456)])
    tl.store($(candidate_vto_copy_50_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(147456))
    r202_value = tl.where(((((r202_z / $(1536)) % $(96)) >= $(0)) & (((r202_z / $(1536)) % $(96)) < $(96)) & ((((r202_z / $(1536)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_50_ptr) + ((((r202_z / $(192)) % $(8)) * $(18432)) + (((((r202_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(192)) + (r202_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r202_value = r202_value.to(tl.bfloat16)
    r202_value = tl.broadcast_to(r202_value, [$(147456)])
    tl.store($(candidate_vslice_backward_1_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(49152))
    r203_value = tl.load($(candidate_vbmm_39_ptr) + ((((r203_z / $(6144)) % $(8)) * $(6144)) + (((r203_z / $(192)) % $(32)) * $(192)) + (r203_z % $(192))), dtype=tl.float32)
    r203_value = r203_value.to(tl.bfloat16)
    r203_value = tl.broadcast_to(r203_value, [$(49152)])
    tl.store($(candidate_vto_copy_51_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(49152))
    r204_value = tl.where(((((r204_z / $(1536)) % $(32)) >= $(0)) & (((r204_z / $(1536)) % $(32)) < $(32)) & ((((r204_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_51_ptr) + ((((r204_z / $(192)) % $(8)) * $(6144)) + (((((r204_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(192)) + (r204_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r204_value = r204_value.to(tl.bfloat16)
    r204_value = tl.broadcast_to(r204_value, [$(49152)])
    tl.store($(candidate_vslice_backward_2_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(0))
    r205_value = tl.cast(0.0, tl.bfloat16)
    r205_value = r205_value.to(tl.bfloat16)
    r205_value = tl.broadcast_to(r205_value, [$(0)])
    tl.store($(candidate_vfill_ptr) + r205_z, r205_value)
    r206_z = tl.arange($(98304))
    r206_value = tl.where(((((r206_z / $(1024)) % $(96)) >= $(96)) & (((r206_z / $(1024)) % $(96)) < $(96)) & ((((r206_z / $(1024)) % $(96)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_ptr) + ((((((r206_z / $(1024)) % $(96)) - $(96)) % $(0)) * $(1024)) + (((r206_z / $(128)) % $(8)) * $(128)) + (r206_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_ptr) + ((((r206_z / $(1024)) % $(96)) * $(1024)) + (((r206_z / $(128)) % $(8)) * $(128)) + (r206_z % $(128))), dtype=tl.bfloat16))
    r206_value = r206_value.to(tl.bfloat16)
    r206_value = tl.broadcast_to(r206_value, [$(98304)])
    tl.store($(candidate_vslice_scatter_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(0))
    r207_value = tl.cast(0.0, tl.bfloat16)
    r207_value = r207_value.to(tl.bfloat16)
    r207_value = tl.broadcast_to(r207_value, [$(0)])
    tl.store($(candidate_vfill_1_ptr) + r207_z, r207_value)
    r208_z = tl.arange($(147456))
    r208_value = tl.where(((((r208_z / $(1536)) % $(96)) >= $(96)) & (((r208_z / $(1536)) % $(96)) < $(96)) & ((((r208_z / $(1536)) % $(96)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_1_ptr) + ((((((r208_z / $(1536)) % $(96)) - $(96)) % $(0)) * $(1536)) + (((r208_z / $(192)) % $(8)) * $(192)) + (r208_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_1_ptr) + ((((r208_z / $(1536)) % $(96)) * $(1536)) + (((r208_z / $(192)) % $(8)) * $(192)) + (r208_z % $(192))), dtype=tl.bfloat16))
    r208_value = r208_value.to(tl.bfloat16)
    r208_value = tl.broadcast_to(r208_value, [$(147456)])
    tl.store($(candidate_vslice_scatter_1_ptr) + r208_z, r208_value)
    r209_z = tl.arange($(6144))
    r209_lane = tl.arange($(8))
    r209_value = tl.sum(tl.load($(candidate_vslice_scatter_1_ptr) + ((((r209_z[:, None] / $(64)) % $(96)) * $(1536)) + (r209_lane[None, :] * $(192)) + (r209_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r209_value = r209_value.to(tl.bfloat16)
    r209_value = tl.broadcast_to(r209_value, [$(6144)])
    tl.store($(candidate_vsum_1_ptr) + r209_z, r209_value)
    r210_row = tl.arange($(96))
    r210_col = tl.arange($(512))
    r210_lane = tl.arange($(1024))
    for r210_group in range($(1)) {
      r210_left = tl.load($(candidate_vslice_scatter_ptr) + ((((((r210_row[:, None] * $(1024)) + r210_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + (((((r210_row[:, None] * $(1024)) + r210_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r210_row[:, None] * $(1024)) + r210_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r210_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r210_col[None, :] * $(1024)) + r210_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r210_col[None, :] * $(1024)) + r210_lane[:, None]) % $(128)) * $(512)) + ((((r210_col[None, :] * $(1024)) + r210_lane[:, None]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r210_value = tl.dot(r210_left, r210_right)
      r210_value = r210_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_40_ptr) + ((r210_group * $(49152)) + (r210_row[:, None] * $(512)) + r210_col[None, :]), r210_value)
    }
    r211_row = tl.arange($(96))
    r211_col = tl.arange($(512))
    r211_lane = tl.arange($(1024))
    for r211_group in range($(1)) {
      r211_left = tl.load($(candidate_vslice_scatter_1_ptr) + ((((((r211_row[:, None] * $(1024)) + r211_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + (((((r211_row[:, None] * $(1024)) + r211_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r211_row[:, None] * $(1024)) + r211_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r211_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r211_col[None, :] * $(1024)) + r211_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r211_col[None, :] * $(1024)) + r211_lane[:, None]) % $(128)) * $(512)) + ((((r211_col[None, :] * $(1024)) + r211_lane[:, None]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r211_value = tl.dot(r211_left, r211_right)
      r211_value = r211_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_41_ptr) + ((r211_group * $(49152)) + (r211_row[:, None] * $(512)) + r211_col[None, :]), r211_value)
    }
    r212_z = tl.arange($(49152))
    r212_value = (tl.load($(candidate_vbmm_40_ptr) + ((((r212_z / $(512)) % $(96)) * $(512)) + (r212_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_41_ptr) + ((((r212_z / $(512)) % $(96)) * $(512)) + (r212_z % $(512))), dtype=tl.bfloat16))
    r212_value = r212_value.to(tl.bfloat16)
    r212_value = tl.broadcast_to(r212_value, [$(49152)])
    tl.store($(candidate_vadd_15_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(49152))
    r213_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_15_ptr) + ((((r213_z / $(512)) % $(96)) * $(512)) + (r213_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r213_value = r213_value.to(tl.bfloat16)
    r213_value = tl.broadcast_to(r213_value, [$(49152)])
    tl.store($(candidate_vselect_backward_ptr) + r213_z, r213_value)
    r214_z = tl.arange($(147456))
    r214_value = tl.where(((((r214_z / $(512)) % $(288)) >= $(192)) & (((r214_z / $(512)) % $(288)) < $(288)) & ((((r214_z / $(512)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_ptr) + ((((((r214_z / $(512)) % $(288)) - $(192)) % $(96)) * $(512)) + (r214_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r214_value = r214_value.to(tl.bfloat16)
    r214_value = tl.broadcast_to(r214_value, [$(147456)])
    tl.store($(candidate_vslice_backward_3_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(18432))
    r215_value = tl.where(((((r215_z / $(64)) % $(288)) >= $(192)) & (((r215_z / $(64)) % $(288)) < $(288)) & ((((r215_z / $(64)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vsum_1_ptr) + ((((((r215_z / $(64)) % $(288)) - $(192)) % $(96)) * $(64)) + (r215_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r215_value = r215_value.to(tl.bfloat16)
    r215_value = tl.broadcast_to(r215_value, [$(18432)])
    tl.store($(candidate_vslice_backward_4_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(294912))
    r216_value = tl.where(((((r216_z / $(1024)) % $(288)) >= $(256)) & (((r216_z / $(1024)) % $(288)) < $(288)) & ((((r216_z / $(1024)) % $(288)) - $(256)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_2_ptr) + ((((((r216_z / $(1024)) % $(288)) - $(256)) % $(32)) * $(1536)) + (((r216_z / $(128)) % $(8)) * $(192)) + (r216_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r216_value = r216_value.to(tl.bfloat16)
    r216_value = tl.broadcast_to(r216_value, [$(294912)])
    tl.store($(candidate_vslice_backward_5_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(147456))
    r217_value = tl.where(((((r217_z / $(512)) % $(288)) >= $(256)) & (((r217_z / $(512)) % $(288)) < $(288)) & ((((r217_z / $(512)) % $(288)) - $(256)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_2_ptr) + ((((((r217_z / $(512)) % $(288)) - $(256)) % $(32)) * $(1536)) + (((r217_z / $(64)) % $(8)) * $(192)) + (r217_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r217_value = r217_value.to(tl.bfloat16)
    r217_value = tl.broadcast_to(r217_value, [$(147456)])
    tl.store($(candidate_vslice_backward_6_ptr) + r217_z, r217_value)
    r218_row = tl.arange($(64))
    r218_col = tl.arange($(128))
    r218_lane = tl.arange($(32))
    for r218_group in range($(8)) {
      r218_left = tl.load($(candidate_vto_copy_45_ptr) + (((((((r218_group % $(8)) * $(2048)) + (r218_lane[None, :] * $(64)) + r218_row[:, None]) / $(2048)) % $(8)) * $(2048)) + ((((((r218_group % $(8)) * $(2048)) + (r218_lane[None, :] * $(64)) + r218_row[:, None]) / $(64)) % $(32)) * $(64)) + ((((r218_group % $(8)) * $(2048)) + (r218_lane[None, :] * $(64)) + r218_row[:, None]) % $(64))), dtype=tl.bfloat16)
      r218_right = tl.load($(candidate_vmm_4_ptr) + (((((((r218_group % $(8)) * $(4096)) + (r218_lane[:, None] * $(128)) + r218_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r218_group % $(8)) * $(4096)) + (r218_lane[:, None] * $(128)) + r218_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r218_group % $(8)) * $(4096)) + (r218_lane[:, None] * $(128)) + r218_col[None, :]) % $(128)) + $(229376)), dtype=tl.bfloat16)
      r218_value = tl.dot(r218_left, r218_right)
      r218_value = r218_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_42_ptr) + ((r218_group * $(8192)) + (r218_row[:, None] * $(128)) + r218_col[None, :]), r218_value)
    }
    r219_row = tl.arange($(32))
    r219_col = tl.arange($(64))
    r219_lane = tl.arange($(128))
    for r219_group in range($(8)) {
      r219_left = tl.load($(candidate_vmm_4_ptr) + (((((((r219_group % $(8)) * $(4096)) + (r219_row[:, None] * $(128)) + r219_lane[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r219_group % $(8)) * $(4096)) + (r219_row[:, None] * $(128)) + r219_lane[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r219_group % $(8)) * $(4096)) + (r219_row[:, None] * $(128)) + r219_lane[None, :]) % $(128)) + $(229376)), dtype=tl.bfloat16)
      r219_right = tl.load($(candidate_vbmm_29_ptr) + (((((((r219_group % $(8)) * $(8192)) + (r219_col[None, :] * $(128)) + r219_lane[:, None]) / $(128)) % $(64)) * $(1024)) + ((((((r219_group % $(8)) * $(8192)) + (r219_col[None, :] * $(128)) + r219_lane[:, None]) / $(8192)) % $(8)) * $(128)) + ((((r219_group % $(8)) * $(8192)) + (r219_col[None, :] * $(128)) + r219_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r219_value = tl.dot(r219_left, r219_right)
      r219_value = r219_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_43_ptr) + ((r219_group * $(2048)) + (r219_row[:, None] * $(64)) + r219_col[None, :]), r219_value)
    }
    r220_z = tl.arange($(16384))
    r220_value = tl.load($(candidate_vbmm_43_ptr) + ((((r220_z / $(2048)) % $(8)) * $(2048)) + (((r220_z / $(64)) % $(32)) * $(64)) + (r220_z % $(64))), dtype=tl.bfloat16)
    r220_value = r220_value.to(tl.float32)
    r220_value = tl.broadcast_to(r220_value, [$(16384)])
    tl.store($(candidate_vto_copy_52_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(16384))
    r221_lane = tl.arange($(64))
    r221_row_values = tl.load($(candidate_vsoftmax_7_ptr) + ((((r221_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r221_z[:, None] / $(64)) % $(32)) * $(64)) + (r221_lane[None, :] % $(64))), dtype=tl.float32)
    r221_row_grad = tl.load($(candidate_vto_copy_52_ptr) + ((((r221_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r221_z[:, None] / $(64)) % $(32)) * $(64)) + (r221_lane[None, :] % $(64))), dtype=tl.float32)
    r221_weighted = r221_row_grad * r221_row_values
    r221_dot_grad = tl.sum(r221_weighted, axis=1)
    r221_value = (tl.load($(candidate_vto_copy_52_ptr) + ((((r221_z / $(2048)) % $(8)) * $(2048)) + (((r221_z / $(64)) % $(32)) * $(64)) + (r221_z % $(64))), dtype=tl.float32) - r221_dot_grad) * tl.load($(candidate_vsoftmax_7_ptr) + ((((r221_z / $(2048)) % $(8)) * $(2048)) + (((r221_z / $(64)) % $(32)) * $(64)) + (r221_z % $(64))), dtype=tl.float32)
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(16384)])
    tl.store($(candidate_vsoftmax_backward_data_1_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(16384))
    r222_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r222_z / $(64)) % $(32)) * $(64)) + (r222_z % $(64))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_1_ptr) + ((((r222_z / $(2048)) % $(8)) * $(2048)) + (((r222_z / $(64)) % $(32)) * $(64)) + (r222_z % $(64))), dtype=tl.float32))
    r222_value = r222_value.to(tl.float32)
    r222_value = tl.broadcast_to(r222_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_10_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(16384))
    r223_value = (tl.load($(candidate_vmasked_fill_10_ptr) + ((((r223_z / $(2048)) % $(8)) * $(2048)) + (((r223_z / $(64)) % $(32)) * $(64)) + (r223_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r223_value = r223_value.to(tl.float32)
    r223_value = tl.broadcast_to(r223_value, [$(16384)])
    tl.store($(candidate_vmul_31_ptr) + r223_z, r223_value)
    r224_row = tl.arange($(192))
    r224_col = tl.arange($(64))
    r224_lane = tl.arange($(32))
    for r224_group in range($(8)) {
      r224_left = tl.load($(candidate_vto_copy_43_ptr) + (((((((r224_group % $(8)) * $(6144)) + (r224_lane[None, :] * $(192)) + r224_row[:, None]) / $(6144)) % $(8)) * $(6144)) + ((((((r224_group % $(8)) * $(6144)) + (r224_lane[None, :] * $(192)) + r224_row[:, None]) / $(192)) % $(32)) * $(192)) + ((((r224_group % $(8)) * $(6144)) + (r224_lane[None, :] * $(192)) + r224_row[:, None]) % $(192))), dtype=tl.float32)
      r224_right = tl.load($(candidate_vmul_31_ptr) + (((((((r224_group % $(8)) * $(2048)) + (r224_lane[:, None] * $(64)) + r224_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r224_group % $(8)) * $(2048)) + (r224_lane[:, None] * $(64)) + r224_col[None, :]) / $(64)) % $(32)) * $(64)) + ((((r224_group % $(8)) * $(2048)) + (r224_lane[:, None] * $(64)) + r224_col[None, :]) % $(64))), dtype=tl.float32)
      r224_value = tl.dot(r224_left, r224_right)
      r224_value = r224_value.to(tl.float32)
      tl.store($(candidate_vbmm_44_ptr) + ((r224_group * $(12288)) + (r224_row[:, None] * $(64)) + r224_col[None, :]), r224_value)
    }
    r225_row = tl.arange($(32))
    r225_col = tl.arange($(192))
    r225_lane = tl.arange($(64))
    for r225_group in range($(8)) {
      r225_left = tl.load($(candidate_vmul_31_ptr) + (((((((r225_group % $(8)) * $(2048)) + (r225_row[:, None] * $(64)) + r225_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r225_group % $(8)) * $(2048)) + (r225_row[:, None] * $(64)) + r225_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r225_group % $(8)) * $(2048)) + (r225_row[:, None] * $(64)) + r225_lane[None, :]) % $(64))), dtype=tl.float32)
      r225_right = tl.load($(candidate_vto_copy_44_ptr) + (((((((r225_group % $(8)) * $(12288)) + (r225_col[None, :] * $(64)) + r225_lane[:, None]) / $(12288)) % $(8)) * $(12288)) + (((((r225_group % $(8)) * $(12288)) + (r225_col[None, :] * $(64)) + r225_lane[:, None]) % $(64)) * $(192)) + (((((r225_group % $(8)) * $(12288)) + (r225_col[None, :] * $(64)) + r225_lane[:, None]) / $(64)) % $(192))), dtype=tl.float32)
      r225_value = tl.dot(r225_left, r225_right)
      r225_value = r225_value.to(tl.float32)
      tl.store($(candidate_vbmm_45_ptr) + ((r225_group * $(6144)) + (r225_row[:, None] * $(192)) + r225_col[None, :]), r225_value)
    }
    r226_z = tl.arange($(98304))
    r226_value = tl.where(((((r226_z / $(1024)) % $(96)) >= $(0)) & (((r226_z / $(1024)) % $(96)) < $(64)) & ((((r226_z / $(1024)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_42_ptr) + ((((((((r226_z / $(128)) % $(8)) * $(8192)) + ((((r226_z / $(1024)) % $(96)) - $(0)) * $(128)) + (r226_z % $(128))) / $(8192)) % $(8)) * $(8192)) + ((((((r226_z / $(128)) % $(8)) * $(64)) + (((r226_z / $(1024)) % $(96)) - $(0))) % $(64)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r226_value = r226_value.to(tl.bfloat16)
    r226_value = tl.broadcast_to(r226_value, [$(98304)])
    tl.store($(candidate_vslice_backward_7_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(98304))
    r227_value = tl.load($(candidate_vbmm_44_ptr) + ((((r227_z / $(12288)) % $(8)) * $(12288)) + ((r227_z % $(192)) * $(64)) + ((r227_z / $(192)) % $(64))), dtype=tl.float32)
    r227_value = r227_value.to(tl.bfloat16)
    r227_value = tl.broadcast_to(r227_value, [$(98304)])
    tl.store($(candidate_vto_copy_53_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(147456))
    r228_value = tl.where(((((r228_z / $(1536)) % $(96)) >= $(0)) & (((r228_z / $(1536)) % $(96)) < $(64)) & ((((r228_z / $(1536)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_53_ptr) + ((((r228_z / $(192)) % $(8)) * $(12288)) + (((((r228_z / $(1536)) % $(96)) - $(0)) % $(64)) * $(192)) + (r228_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r228_value = r228_value.to(tl.bfloat16)
    r228_value = tl.broadcast_to(r228_value, [$(147456)])
    tl.store($(candidate_vslice_backward_8_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(49152))
    r229_value = tl.load($(candidate_vbmm_45_ptr) + ((((r229_z / $(6144)) % $(8)) * $(6144)) + (((r229_z / $(192)) % $(32)) * $(192)) + (r229_z % $(192))), dtype=tl.float32)
    r229_value = r229_value.to(tl.bfloat16)
    r229_value = tl.broadcast_to(r229_value, [$(49152)])
    tl.store($(candidate_vto_copy_54_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(49152))
    r230_value = tl.where(((((r230_z / $(1536)) % $(32)) >= $(0)) & (((r230_z / $(1536)) % $(32)) < $(32)) & ((((r230_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_54_ptr) + ((((r230_z / $(192)) % $(8)) * $(6144)) + (((((r230_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(192)) + (r230_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r230_value = r230_value.to(tl.bfloat16)
    r230_value = tl.broadcast_to(r230_value, [$(49152)])
    tl.store($(candidate_vslice_backward_9_ptr) + r230_z, r230_value)
    r231_z = tl.arange($(32768))
    r231_value = tl.cast(0.0, tl.bfloat16)
    r231_value = r231_value.to(tl.bfloat16)
    r231_value = tl.broadcast_to(r231_value, [$(32768)])
    tl.store($(candidate_vfill_2_ptr) + r231_z, r231_value)
    r232_z = tl.arange($(98304))
    r232_value = tl.where(((((r232_z / $(1024)) % $(96)) >= $(64)) & (((r232_z / $(1024)) % $(96)) < $(96)) & ((((r232_z / $(1024)) % $(96)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_2_ptr) + ((((((r232_z / $(1024)) % $(96)) - $(64)) % $(32)) * $(1024)) + (((r232_z / $(128)) % $(8)) * $(128)) + (r232_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_7_ptr) + ((((r232_z / $(1024)) % $(96)) * $(1024)) + (((r232_z / $(128)) % $(8)) * $(128)) + (r232_z % $(128))), dtype=tl.bfloat16))
    r232_value = r232_value.to(tl.bfloat16)
    r232_value = tl.broadcast_to(r232_value, [$(98304)])
    tl.store($(candidate_vslice_scatter_2_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(49152))
    r233_value = tl.cast(0.0, tl.bfloat16)
    r233_value = r233_value.to(tl.bfloat16)
    r233_value = tl.broadcast_to(r233_value, [$(49152)])
    tl.store($(candidate_vfill_3_ptr) + r233_z, r233_value)
    r234_z = tl.arange($(147456))
    r234_value = tl.where(((((r234_z / $(1536)) % $(96)) >= $(64)) & (((r234_z / $(1536)) % $(96)) < $(96)) & ((((r234_z / $(1536)) % $(96)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_3_ptr) + ((((((r234_z / $(1536)) % $(96)) - $(64)) % $(32)) * $(1536)) + (((r234_z / $(192)) % $(8)) * $(192)) + (r234_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_8_ptr) + ((((r234_z / $(1536)) % $(96)) * $(1536)) + (((r234_z / $(192)) % $(8)) * $(192)) + (r234_z % $(192))), dtype=tl.bfloat16))
    r234_value = r234_value.to(tl.bfloat16)
    r234_value = tl.broadcast_to(r234_value, [$(147456)])
    tl.store($(candidate_vslice_scatter_3_ptr) + r234_z, r234_value)
    r235_z = tl.arange($(6144))
    r235_lane = tl.arange($(8))
    r235_value = tl.sum(tl.load($(candidate_vslice_scatter_3_ptr) + ((((r235_z[:, None] / $(64)) % $(96)) * $(1536)) + (r235_lane[None, :] * $(192)) + (r235_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r235_value = r235_value.to(tl.bfloat16)
    r235_value = tl.broadcast_to(r235_value, [$(6144)])
    tl.store($(candidate_vsum_2_ptr) + r235_z, r235_value)
    r236_row = tl.arange($(96))
    r236_col = tl.arange($(512))
    r236_lane = tl.arange($(1024))
    for r236_group in range($(1)) {
      r236_left = tl.load($(candidate_vslice_scatter_2_ptr) + ((((((r236_row[:, None] * $(1024)) + r236_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + (((((r236_row[:, None] * $(1024)) + r236_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r236_row[:, None] * $(1024)) + r236_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r236_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r236_col[None, :] * $(1024)) + r236_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r236_col[None, :] * $(1024)) + r236_lane[:, None]) % $(128)) * $(512)) + ((((r236_col[None, :] * $(1024)) + r236_lane[:, None]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r236_value = tl.dot(r236_left, r236_right)
      r236_value = r236_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_46_ptr) + ((r236_group * $(49152)) + (r236_row[:, None] * $(512)) + r236_col[None, :]), r236_value)
    }
    r237_row = tl.arange($(96))
    r237_col = tl.arange($(512))
    r237_lane = tl.arange($(1024))
    for r237_group in range($(1)) {
      r237_left = tl.load($(candidate_vslice_scatter_3_ptr) + ((((((r237_row[:, None] * $(1024)) + r237_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + (((((r237_row[:, None] * $(1024)) + r237_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r237_row[:, None] * $(1024)) + r237_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r237_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r237_col[None, :] * $(1024)) + r237_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r237_col[None, :] * $(1024)) + r237_lane[:, None]) % $(128)) * $(512)) + ((((r237_col[None, :] * $(1024)) + r237_lane[:, None]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r237_value = tl.dot(r237_left, r237_right)
      r237_value = r237_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_47_ptr) + ((r237_group * $(49152)) + (r237_row[:, None] * $(512)) + r237_col[None, :]), r237_value)
    }
    r238_z = tl.arange($(49152))
    r238_value = (tl.load($(candidate_vbmm_46_ptr) + ((((r238_z / $(512)) % $(96)) * $(512)) + (r238_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_47_ptr) + ((((r238_z / $(512)) % $(96)) * $(512)) + (r238_z % $(512))), dtype=tl.bfloat16))
    r238_value = r238_value.to(tl.bfloat16)
    r238_value = tl.broadcast_to(r238_value, [$(49152)])
    tl.store($(candidate_vadd_16_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(49152))
    r239_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_16_ptr) + ((((r239_z / $(512)) % $(96)) * $(512)) + (r239_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r239_value = r239_value.to(tl.bfloat16)
    r239_value = tl.broadcast_to(r239_value, [$(49152)])
    tl.store($(candidate_vselect_backward_1_ptr) + r239_z, r239_value)
    r240_z = tl.arange($(147456))
    r240_value = tl.where(((((r240_z / $(512)) % $(288)) >= $(192)) & (((r240_z / $(512)) % $(288)) < $(288)) & ((((r240_z / $(512)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_1_ptr) + ((((((r240_z / $(512)) % $(288)) - $(192)) % $(96)) * $(512)) + (r240_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r240_value = r240_value.to(tl.bfloat16)
    r240_value = tl.broadcast_to(r240_value, [$(147456)])
    tl.store($(candidate_vslice_backward_10_ptr) + r240_z, r240_value)
    r241_z = tl.arange($(147456))
    r241_value = (tl.load($(candidate_vslice_backward_3_ptr) + ((((r241_z / $(512)) % $(288)) * $(512)) + (r241_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_10_ptr) + ((((r241_z / $(512)) % $(288)) * $(512)) + (r241_z % $(512))), dtype=tl.bfloat16))
    r241_value = r241_value.to(tl.bfloat16)
    r241_value = tl.broadcast_to(r241_value, [$(147456)])
    tl.store($(candidate_vadd_17_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(18432))
    r242_value = tl.where(((((r242_z / $(64)) % $(288)) >= $(192)) & (((r242_z / $(64)) % $(288)) < $(288)) & ((((r242_z / $(64)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vsum_2_ptr) + ((((((r242_z / $(64)) % $(288)) - $(192)) % $(96)) * $(64)) + (r242_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r242_value = r242_value.to(tl.bfloat16)
    r242_value = tl.broadcast_to(r242_value, [$(18432)])
    tl.store($(candidate_vslice_backward_11_ptr) + r242_z, r242_value)
    r243_z = tl.arange($(18432))
    r243_value = (tl.load($(candidate_vslice_backward_4_ptr) + ((((r243_z / $(64)) % $(288)) * $(64)) + (r243_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_11_ptr) + ((((r243_z / $(64)) % $(288)) * $(64)) + (r243_z % $(64))), dtype=tl.bfloat16))
    r243_value = r243_value.to(tl.bfloat16)
    r243_value = tl.broadcast_to(r243_value, [$(18432)])
    tl.store($(candidate_vadd_18_ptr) + r243_z, r243_value)
    r244_z = tl.arange($(294912))
    r244_value = tl.where(((((r244_z / $(1024)) % $(288)) >= $(224)) & (((r244_z / $(1024)) % $(288)) < $(256)) & ((((r244_z / $(1024)) % $(288)) - $(224)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_9_ptr) + ((((((r244_z / $(1024)) % $(288)) - $(224)) % $(32)) * $(1536)) + (((r244_z / $(128)) % $(8)) * $(192)) + (r244_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r244_value = r244_value.to(tl.bfloat16)
    r244_value = tl.broadcast_to(r244_value, [$(294912)])
    tl.store($(candidate_vslice_backward_12_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(294912))
    r245_value = (tl.load($(candidate_vslice_backward_5_ptr) + ((((r245_z / $(1024)) % $(288)) * $(1024)) + (((r245_z / $(128)) % $(8)) * $(128)) + (r245_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_12_ptr) + ((((r245_z / $(1024)) % $(288)) * $(1024)) + (((r245_z / $(128)) % $(8)) * $(128)) + (r245_z % $(128))), dtype=tl.bfloat16))
    r245_value = r245_value.to(tl.bfloat16)
    r245_value = tl.broadcast_to(r245_value, [$(294912)])
    tl.store($(candidate_vadd_19_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(147456))
    r246_value = tl.where(((((r246_z / $(512)) % $(288)) >= $(224)) & (((r246_z / $(512)) % $(288)) < $(256)) & ((((r246_z / $(512)) % $(288)) - $(224)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_9_ptr) + ((((((r246_z / $(512)) % $(288)) - $(224)) % $(32)) * $(1536)) + (((r246_z / $(64)) % $(8)) * $(192)) + (r246_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r246_value = r246_value.to(tl.bfloat16)
    r246_value = tl.broadcast_to(r246_value, [$(147456)])
    tl.store($(candidate_vslice_backward_13_ptr) + r246_z, r246_value)
    r247_z = tl.arange($(147456))
    r247_value = (tl.load($(candidate_vslice_backward_6_ptr) + ((((r247_z / $(512)) % $(288)) * $(512)) + (((r247_z / $(64)) % $(8)) * $(64)) + (r247_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_13_ptr) + ((((r247_z / $(512)) % $(288)) * $(512)) + (((r247_z / $(64)) % $(8)) * $(64)) + (r247_z % $(64))), dtype=tl.bfloat16))
    r247_value = r247_value.to(tl.bfloat16)
    r247_value = tl.broadcast_to(r247_value, [$(147456)])
    tl.store($(candidate_vadd_20_ptr) + r247_z, r247_value)
    r248_row = tl.arange($(32))
    r248_col = tl.arange($(128))
    r248_lane = tl.arange($(32))
    for r248_group in range($(8)) {
      r248_left = tl.load($(candidate_vto_copy_42_ptr) + (((((((r248_group % $(8)) * $(1024)) + (r248_lane[None, :] * $(32)) + r248_row[:, None]) / $(1024)) % $(8)) * $(1024)) + ((((((r248_group % $(8)) * $(1024)) + (r248_lane[None, :] * $(32)) + r248_row[:, None]) / $(32)) % $(32)) * $(32)) + ((((r248_group % $(8)) * $(1024)) + (r248_lane[None, :] * $(32)) + r248_row[:, None]) % $(32))), dtype=tl.bfloat16)
      r248_right = tl.load($(candidate_vmm_4_ptr) + (((((((r248_group % $(8)) * $(4096)) + (r248_lane[:, None] * $(128)) + r248_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r248_group % $(8)) * $(4096)) + (r248_lane[:, None] * $(128)) + r248_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r248_group % $(8)) * $(4096)) + (r248_lane[:, None] * $(128)) + r248_col[None, :]) % $(128)) + $(196608)), dtype=tl.bfloat16)
      r248_value = tl.dot(r248_left, r248_right)
      r248_value = r248_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_48_ptr) + ((r248_group * $(4096)) + (r248_row[:, None] * $(128)) + r248_col[None, :]), r248_value)
    }
    r249_row = tl.arange($(32))
    r249_col = tl.arange($(32))
    r249_lane = tl.arange($(128))
    for r249_group in range($(8)) {
      r249_left = tl.load($(candidate_vmm_4_ptr) + (((((((r249_group % $(8)) * $(4096)) + (r249_row[:, None] * $(128)) + r249_lane[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r249_group % $(8)) * $(4096)) + (r249_row[:, None] * $(128)) + r249_lane[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r249_group % $(8)) * $(4096)) + (r249_row[:, None] * $(128)) + r249_lane[None, :]) % $(128)) + $(196608)), dtype=tl.bfloat16)
      r249_right = tl.load($(candidate_vbmm_25_ptr) + (((((((r249_group % $(8)) * $(4096)) + (r249_col[None, :] * $(128)) + r249_lane[:, None]) / $(128)) % $(32)) * $(1024)) + ((((((r249_group % $(8)) * $(4096)) + (r249_col[None, :] * $(128)) + r249_lane[:, None]) / $(4096)) % $(8)) * $(128)) + ((((r249_group % $(8)) * $(4096)) + (r249_col[None, :] * $(128)) + r249_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r249_value = tl.dot(r249_left, r249_right)
      r249_value = r249_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_49_ptr) + ((r249_group * $(1024)) + (r249_row[:, None] * $(32)) + r249_col[None, :]), r249_value)
    }
    r250_z = tl.arange($(8192))
    r250_value = tl.load($(candidate_vbmm_49_ptr) + ((((r250_z / $(1024)) % $(8)) * $(1024)) + (((r250_z / $(32)) % $(32)) * $(32)) + (r250_z % $(32))), dtype=tl.bfloat16)
    r250_value = r250_value.to(tl.float32)
    r250_value = tl.broadcast_to(r250_value, [$(8192)])
    tl.store($(candidate_vto_copy_55_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(8192))
    r251_lane = tl.arange($(32))
    r251_row_values = tl.load($(candidate_vsoftmax_6_ptr) + ((((r251_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r251_z[:, None] / $(32)) % $(32)) * $(32)) + (r251_lane[None, :] % $(32))), dtype=tl.float32)
    r251_row_grad = tl.load($(candidate_vto_copy_55_ptr) + ((((r251_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r251_z[:, None] / $(32)) % $(32)) * $(32)) + (r251_lane[None, :] % $(32))), dtype=tl.float32)
    r251_weighted = r251_row_grad * r251_row_values
    r251_dot_grad = tl.sum(r251_weighted, axis=1)
    r251_value = (tl.load($(candidate_vto_copy_55_ptr) + ((((r251_z / $(1024)) % $(8)) * $(1024)) + (((r251_z / $(32)) % $(32)) * $(32)) + (r251_z % $(32))), dtype=tl.float32) - r251_dot_grad) * tl.load($(candidate_vsoftmax_6_ptr) + ((((r251_z / $(1024)) % $(8)) * $(1024)) + (((r251_z / $(32)) % $(32)) * $(32)) + (r251_z % $(32))), dtype=tl.float32)
    r251_value = r251_value.to(tl.float32)
    r251_value = tl.broadcast_to(r251_value, [$(8192)])
    tl.store($(candidate_vsoftmax_backward_data_2_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(8192))
    r252_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r252_z / $(32)) % $(32)) * $(32)) + (r252_z % $(32))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_2_ptr) + ((((r252_z / $(1024)) % $(8)) * $(1024)) + (((r252_z / $(32)) % $(32)) * $(32)) + (r252_z % $(32))), dtype=tl.float32))
    r252_value = r252_value.to(tl.float32)
    r252_value = tl.broadcast_to(r252_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_11_ptr) + r252_z, r252_value)
    r253_z = tl.arange($(8192))
    r253_value = (tl.load($(candidate_vmasked_fill_11_ptr) + ((((r253_z / $(1024)) % $(8)) * $(1024)) + (((r253_z / $(32)) % $(32)) * $(32)) + (r253_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r253_value = r253_value.to(tl.float32)
    r253_value = tl.broadcast_to(r253_value, [$(8192)])
    tl.store($(candidate_vmul_32_ptr) + r253_z, r253_value)
    r254_row = tl.arange($(192))
    r254_col = tl.arange($(32))
    r254_lane = tl.arange($(32))
    for r254_group in range($(8)) {
      r254_left = tl.load($(candidate_vto_copy_40_ptr) + (((((((r254_group % $(8)) * $(6144)) + (r254_lane[None, :] * $(192)) + r254_row[:, None]) / $(6144)) % $(8)) * $(6144)) + ((((((r254_group % $(8)) * $(6144)) + (r254_lane[None, :] * $(192)) + r254_row[:, None]) / $(192)) % $(32)) * $(192)) + ((((r254_group % $(8)) * $(6144)) + (r254_lane[None, :] * $(192)) + r254_row[:, None]) % $(192))), dtype=tl.float32)
      r254_right = tl.load($(candidate_vmul_32_ptr) + (((((((r254_group % $(8)) * $(1024)) + (r254_lane[:, None] * $(32)) + r254_col[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r254_group % $(8)) * $(1024)) + (r254_lane[:, None] * $(32)) + r254_col[None, :]) / $(32)) % $(32)) * $(32)) + ((((r254_group % $(8)) * $(1024)) + (r254_lane[:, None] * $(32)) + r254_col[None, :]) % $(32))), dtype=tl.float32)
      r254_value = tl.dot(r254_left, r254_right)
      r254_value = r254_value.to(tl.float32)
      tl.store($(candidate_vbmm_50_ptr) + ((r254_group * $(6144)) + (r254_row[:, None] * $(32)) + r254_col[None, :]), r254_value)
    }
    r255_row = tl.arange($(32))
    r255_col = tl.arange($(192))
    r255_lane = tl.arange($(32))
    for r255_group in range($(8)) {
      r255_left = tl.load($(candidate_vmul_32_ptr) + (((((((r255_group % $(8)) * $(1024)) + (r255_row[:, None] * $(32)) + r255_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r255_group % $(8)) * $(1024)) + (r255_row[:, None] * $(32)) + r255_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r255_group % $(8)) * $(1024)) + (r255_row[:, None] * $(32)) + r255_lane[None, :]) % $(32))), dtype=tl.float32)
      r255_right = tl.load($(candidate_vto_copy_41_ptr) + (((((((r255_group % $(8)) * $(6144)) + (r255_col[None, :] * $(32)) + r255_lane[:, None]) / $(6144)) % $(8)) * $(6144)) + (((((r255_group % $(8)) * $(6144)) + (r255_col[None, :] * $(32)) + r255_lane[:, None]) % $(32)) * $(192)) + (((((r255_group % $(8)) * $(6144)) + (r255_col[None, :] * $(32)) + r255_lane[:, None]) / $(32)) % $(192))), dtype=tl.float32)
      r255_value = tl.dot(r255_left, r255_right)
      r255_value = r255_value.to(tl.float32)
      tl.store($(candidate_vbmm_51_ptr) + ((r255_group * $(6144)) + (r255_row[:, None] * $(192)) + r255_col[None, :]), r255_value)
    }
    r256_z = tl.arange($(98304))
    r256_value = tl.where(((((r256_z / $(1024)) % $(96)) >= $(0)) & (((r256_z / $(1024)) % $(96)) < $(32)) & ((((r256_z / $(1024)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_48_ptr) + ((((((((r256_z / $(128)) % $(8)) * $(4096)) + ((((r256_z / $(1024)) % $(96)) - $(0)) * $(128)) + (r256_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r256_z / $(128)) % $(8)) * $(32)) + (((r256_z / $(1024)) % $(96)) - $(0))) % $(32)) * $(128)) + (r256_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r256_value = r256_value.to(tl.bfloat16)
    r256_value = tl.broadcast_to(r256_value, [$(98304)])
    tl.store($(candidate_vslice_backward_14_ptr) + r256_z, r256_value)
    r257_z = tl.arange($(49152))
    r257_value = tl.load($(candidate_vbmm_50_ptr) + ((((r257_z / $(6144)) % $(8)) * $(6144)) + ((r257_z % $(192)) * $(32)) + ((r257_z / $(192)) % $(32))), dtype=tl.float32)
    r257_value = r257_value.to(tl.bfloat16)
    r257_value = tl.broadcast_to(r257_value, [$(49152)])
    tl.store($(candidate_vto_copy_56_ptr) + r257_z, r257_value)
    r258_z = tl.arange($(147456))
    r258_value = tl.where(((((r258_z / $(1536)) % $(96)) >= $(0)) & (((r258_z / $(1536)) % $(96)) < $(32)) & ((((r258_z / $(1536)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_56_ptr) + ((((r258_z / $(192)) % $(8)) * $(6144)) + (((((r258_z / $(1536)) % $(96)) - $(0)) % $(32)) * $(192)) + (r258_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r258_value = r258_value.to(tl.bfloat16)
    r258_value = tl.broadcast_to(r258_value, [$(147456)])
    tl.store($(candidate_vslice_backward_15_ptr) + r258_z, r258_value)
    r259_z = tl.arange($(49152))
    r259_value = tl.load($(candidate_vbmm_51_ptr) + ((((r259_z / $(6144)) % $(8)) * $(6144)) + (((r259_z / $(192)) % $(32)) * $(192)) + (r259_z % $(192))), dtype=tl.float32)
    r259_value = r259_value.to(tl.bfloat16)
    r259_value = tl.broadcast_to(r259_value, [$(49152)])
    tl.store($(candidate_vto_copy_57_ptr) + r259_z, r259_value)
    r260_z = tl.arange($(49152))
    r260_value = tl.where(((((r260_z / $(1536)) % $(32)) >= $(0)) & (((r260_z / $(1536)) % $(32)) < $(32)) & ((((r260_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_57_ptr) + ((((r260_z / $(192)) % $(8)) * $(6144)) + (((((r260_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(192)) + (r260_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r260_value = r260_value.to(tl.bfloat16)
    r260_value = tl.broadcast_to(r260_value, [$(49152)])
    tl.store($(candidate_vslice_backward_16_ptr) + r260_z, r260_value)
    r261_z = tl.arange($(65536))
    r261_value = tl.cast(0.0, tl.bfloat16)
    r261_value = r261_value.to(tl.bfloat16)
    r261_value = tl.broadcast_to(r261_value, [$(65536)])
    tl.store($(candidate_vfill_4_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(98304))
    r262_value = tl.where(((((r262_z / $(1024)) % $(96)) >= $(32)) & (((r262_z / $(1024)) % $(96)) < $(96)) & ((((r262_z / $(1024)) % $(96)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vfill_4_ptr) + ((((((r262_z / $(1024)) % $(96)) - $(32)) % $(64)) * $(1024)) + (((r262_z / $(128)) % $(8)) * $(128)) + (r262_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_14_ptr) + ((((r262_z / $(1024)) % $(96)) * $(1024)) + (((r262_z / $(128)) % $(8)) * $(128)) + (r262_z % $(128))), dtype=tl.bfloat16))
    r262_value = r262_value.to(tl.bfloat16)
    r262_value = tl.broadcast_to(r262_value, [$(98304)])
    tl.store($(candidate_vslice_scatter_4_ptr) + r262_z, r262_value)
    r263_z = tl.arange($(98304))
    r263_value = tl.cast(0.0, tl.bfloat16)
    r263_value = r263_value.to(tl.bfloat16)
    r263_value = tl.broadcast_to(r263_value, [$(98304)])
    tl.store($(candidate_vfill_5_ptr) + r263_z, r263_value)
    r264_z = tl.arange($(147456))
    r264_value = tl.where(((((r264_z / $(1536)) % $(96)) >= $(32)) & (((r264_z / $(1536)) % $(96)) < $(96)) & ((((r264_z / $(1536)) % $(96)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vfill_5_ptr) + ((((((r264_z / $(1536)) % $(96)) - $(32)) % $(64)) * $(1536)) + (((r264_z / $(192)) % $(8)) * $(192)) + (r264_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_15_ptr) + ((((r264_z / $(1536)) % $(96)) * $(1536)) + (((r264_z / $(192)) % $(8)) * $(192)) + (r264_z % $(192))), dtype=tl.bfloat16))
    r264_value = r264_value.to(tl.bfloat16)
    r264_value = tl.broadcast_to(r264_value, [$(147456)])
    tl.store($(candidate_vslice_scatter_5_ptr) + r264_z, r264_value)
    r265_z = tl.arange($(6144))
    r265_lane = tl.arange($(8))
    r265_value = tl.sum(tl.load($(candidate_vslice_scatter_5_ptr) + ((((r265_z[:, None] / $(64)) % $(96)) * $(1536)) + (r265_lane[None, :] * $(192)) + (r265_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r265_value = r265_value.to(tl.bfloat16)
    r265_value = tl.broadcast_to(r265_value, [$(6144)])
    tl.store($(candidate_vsum_3_ptr) + r265_z, r265_value)
    r266_row = tl.arange($(96))
    r266_col = tl.arange($(512))
    r266_lane = tl.arange($(1024))
    for r266_group in range($(1)) {
      r266_left = tl.load($(candidate_vslice_scatter_4_ptr) + ((((((r266_row[:, None] * $(1024)) + r266_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + (((((r266_row[:, None] * $(1024)) + r266_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r266_row[:, None] * $(1024)) + r266_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r266_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r266_col[None, :] * $(1024)) + r266_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r266_col[None, :] * $(1024)) + r266_lane[:, None]) % $(128)) * $(512)) + ((((r266_col[None, :] * $(1024)) + r266_lane[:, None]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r266_value = tl.dot(r266_left, r266_right)
      r266_value = r266_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_52_ptr) + ((r266_group * $(49152)) + (r266_row[:, None] * $(512)) + r266_col[None, :]), r266_value)
    }
    r267_row = tl.arange($(96))
    r267_col = tl.arange($(512))
    r267_lane = tl.arange($(1024))
    for r267_group in range($(1)) {
      r267_left = tl.load($(candidate_vslice_scatter_5_ptr) + ((((((r267_row[:, None] * $(1024)) + r267_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + (((((r267_row[:, None] * $(1024)) + r267_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r267_row[:, None] * $(1024)) + r267_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r267_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r267_col[None, :] * $(1024)) + r267_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r267_col[None, :] * $(1024)) + r267_lane[:, None]) % $(128)) * $(512)) + ((((r267_col[None, :] * $(1024)) + r267_lane[:, None]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r267_value = tl.dot(r267_left, r267_right)
      r267_value = r267_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_53_ptr) + ((r267_group * $(49152)) + (r267_row[:, None] * $(512)) + r267_col[None, :]), r267_value)
    }
    r268_z = tl.arange($(49152))
    r268_value = (tl.load($(candidate_vbmm_52_ptr) + ((((r268_z / $(512)) % $(96)) * $(512)) + (r268_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_53_ptr) + ((((r268_z / $(512)) % $(96)) * $(512)) + (r268_z % $(512))), dtype=tl.bfloat16))
    r268_value = r268_value.to(tl.bfloat16)
    r268_value = tl.broadcast_to(r268_value, [$(49152)])
    tl.store($(candidate_vadd_21_ptr) + r268_z, r268_value)
    r269_z = tl.arange($(49152))
    r269_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_21_ptr) + ((((r269_z / $(512)) % $(96)) * $(512)) + (r269_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r269_value = r269_value.to(tl.bfloat16)
    r269_value = tl.broadcast_to(r269_value, [$(49152)])
    tl.store($(candidate_vselect_backward_2_ptr) + r269_z, r269_value)
    r270_z = tl.arange($(147456))
    r270_value = tl.where(((((r270_z / $(512)) % $(288)) >= $(192)) & (((r270_z / $(512)) % $(288)) < $(288)) & ((((r270_z / $(512)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_2_ptr) + ((((((r270_z / $(512)) % $(288)) - $(192)) % $(96)) * $(512)) + (r270_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r270_value = r270_value.to(tl.bfloat16)
    r270_value = tl.broadcast_to(r270_value, [$(147456)])
    tl.store($(candidate_vslice_backward_17_ptr) + r270_z, r270_value)
    r271_z = tl.arange($(147456))
    r271_value = (tl.load($(candidate_vadd_17_ptr) + ((((r271_z / $(512)) % $(288)) * $(512)) + (r271_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_17_ptr) + ((((r271_z / $(512)) % $(288)) * $(512)) + (r271_z % $(512))), dtype=tl.bfloat16))
    r271_value = r271_value.to(tl.bfloat16)
    r271_value = tl.broadcast_to(r271_value, [$(147456)])
    tl.store($(candidate_vadd_22_ptr) + r271_z, r271_value)
    r272_z = tl.arange($(18432))
    r272_value = tl.where(((((r272_z / $(64)) % $(288)) >= $(192)) & (((r272_z / $(64)) % $(288)) < $(288)) & ((((r272_z / $(64)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vsum_3_ptr) + ((((((r272_z / $(64)) % $(288)) - $(192)) % $(96)) * $(64)) + (r272_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r272_value = r272_value.to(tl.bfloat16)
    r272_value = tl.broadcast_to(r272_value, [$(18432)])
    tl.store($(candidate_vslice_backward_18_ptr) + r272_z, r272_value)
    r273_z = tl.arange($(18432))
    r273_value = (tl.load($(candidate_vadd_18_ptr) + ((((r273_z / $(64)) % $(288)) * $(64)) + (r273_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_18_ptr) + ((((r273_z / $(64)) % $(288)) * $(64)) + (r273_z % $(64))), dtype=tl.bfloat16))
    r273_value = r273_value.to(tl.bfloat16)
    r273_value = tl.broadcast_to(r273_value, [$(18432)])
    tl.store($(candidate_vadd_23_ptr) + r273_z, r273_value)
    r274_z = tl.arange($(294912))
    r274_value = tl.where(((((r274_z / $(1024)) % $(288)) >= $(192)) & (((r274_z / $(1024)) % $(288)) < $(224)) & ((((r274_z / $(1024)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_16_ptr) + ((((((r274_z / $(1024)) % $(288)) - $(192)) % $(32)) * $(1536)) + (((r274_z / $(128)) % $(8)) * $(192)) + (r274_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r274_value = r274_value.to(tl.bfloat16)
    r274_value = tl.broadcast_to(r274_value, [$(294912)])
    tl.store($(candidate_vslice_backward_19_ptr) + r274_z, r274_value)
    r275_z = tl.arange($(294912))
    r275_value = (tl.load($(candidate_vadd_19_ptr) + ((((r275_z / $(1024)) % $(288)) * $(1024)) + (((r275_z / $(128)) % $(8)) * $(128)) + (r275_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_19_ptr) + ((((r275_z / $(1024)) % $(288)) * $(1024)) + (((r275_z / $(128)) % $(8)) * $(128)) + (r275_z % $(128))), dtype=tl.bfloat16))
    r275_value = r275_value.to(tl.bfloat16)
    r275_value = tl.broadcast_to(r275_value, [$(294912)])
    tl.store($(candidate_vadd_24_ptr) + r275_z, r275_value)
    r276_z = tl.arange($(147456))
    r276_value = tl.where(((((r276_z / $(512)) % $(288)) >= $(192)) & (((r276_z / $(512)) % $(288)) < $(224)) & ((((r276_z / $(512)) % $(288)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_16_ptr) + ((((((r276_z / $(512)) % $(288)) - $(192)) % $(32)) * $(1536)) + (((r276_z / $(64)) % $(8)) * $(192)) + (r276_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r276_value = r276_value.to(tl.bfloat16)
    r276_value = tl.broadcast_to(r276_value, [$(147456)])
    tl.store($(candidate_vslice_backward_20_ptr) + r276_z, r276_value)
    r277_z = tl.arange($(147456))
    r277_value = (tl.load($(candidate_vadd_20_ptr) + ((((r277_z / $(512)) % $(288)) * $(512)) + (((r277_z / $(64)) % $(8)) * $(64)) + (r277_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_20_ptr) + ((((r277_z / $(512)) % $(288)) * $(512)) + (((r277_z / $(64)) % $(8)) * $(64)) + (r277_z % $(64))), dtype=tl.bfloat16))
    r277_value = r277_value.to(tl.bfloat16)
    r277_value = tl.broadcast_to(r277_value, [$(147456)])
    tl.store($(candidate_vadd_25_ptr) + r277_z, r277_value)
    r278_row = tl.arange($(96))
    r278_col = tl.arange($(128))
    r278_lane = tl.arange($(32))
    for r278_group in range($(8)) {
      r278_left = tl.load($(candidate_vto_copy_39_ptr) + (((((((r278_group % $(8)) * $(3072)) + (r278_lane[None, :] * $(96)) + r278_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r278_group % $(8)) * $(3072)) + (r278_lane[None, :] * $(96)) + r278_row[:, None]) / $(96)) % $(32)) * $(96)) + ((((r278_group % $(8)) * $(3072)) + (r278_lane[None, :] * $(96)) + r278_row[:, None]) % $(96))), dtype=tl.bfloat16)
      r278_right = tl.load($(candidate_vmm_4_ptr) + (((((((r278_group % $(8)) * $(4096)) + (r278_lane[:, None] * $(128)) + r278_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r278_group % $(8)) * $(4096)) + (r278_lane[:, None] * $(128)) + r278_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r278_group % $(8)) * $(4096)) + (r278_lane[:, None] * $(128)) + r278_col[None, :]) % $(128)) + $(163840)), dtype=tl.bfloat16)
      r278_value = tl.dot(r278_left, r278_right)
      r278_value = r278_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_54_ptr) + ((r278_group * $(12288)) + (r278_row[:, None] * $(128)) + r278_col[None, :]), r278_value)
    }
    r279_row = tl.arange($(32))
    r279_col = tl.arange($(96))
    r279_lane = tl.arange($(128))
    for r279_group in range($(8)) {
      r279_left = tl.load($(candidate_vmm_4_ptr) + (((((((r279_group % $(8)) * $(4096)) + (r279_row[:, None] * $(128)) + r279_lane[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r279_group % $(8)) * $(4096)) + (r279_row[:, None] * $(128)) + r279_lane[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r279_group % $(8)) * $(4096)) + (r279_row[:, None] * $(128)) + r279_lane[None, :]) % $(128)) + $(163840)), dtype=tl.bfloat16)
      r279_right = tl.load($(candidate_vbmm_21_ptr) + (((((((r279_group % $(8)) * $(12288)) + (r279_col[None, :] * $(128)) + r279_lane[:, None]) / $(128)) % $(96)) * $(1024)) + ((((((r279_group % $(8)) * $(12288)) + (r279_col[None, :] * $(128)) + r279_lane[:, None]) / $(12288)) % $(8)) * $(128)) + ((((r279_group % $(8)) * $(12288)) + (r279_col[None, :] * $(128)) + r279_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r279_value = tl.dot(r279_left, r279_right)
      r279_value = r279_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_55_ptr) + ((r279_group * $(3072)) + (r279_row[:, None] * $(96)) + r279_col[None, :]), r279_value)
    }
    r280_z = tl.arange($(24576))
    r280_value = tl.load($(candidate_vbmm_55_ptr) + ((((r280_z / $(3072)) % $(8)) * $(3072)) + (((r280_z / $(96)) % $(32)) * $(96)) + (r280_z % $(96))), dtype=tl.bfloat16)
    r280_value = r280_value.to(tl.float32)
    r280_value = tl.broadcast_to(r280_value, [$(24576)])
    tl.store($(candidate_vto_copy_58_ptr) + r280_z, r280_value)
    r281_z = tl.arange($(24576))
    r281_lane = tl.arange($(96))
    r281_row_values = tl.load($(candidate_vsoftmax_5_ptr) + ((((r281_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r281_z[:, None] / $(96)) % $(32)) * $(96)) + (r281_lane[None, :] % $(96))), dtype=tl.float32)
    r281_row_grad = tl.load($(candidate_vto_copy_58_ptr) + ((((r281_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r281_z[:, None] / $(96)) % $(32)) * $(96)) + (r281_lane[None, :] % $(96))), dtype=tl.float32)
    r281_weighted = r281_row_grad * r281_row_values
    r281_dot_grad = tl.sum(r281_weighted, axis=1)
    r281_value = (tl.load($(candidate_vto_copy_58_ptr) + ((((r281_z / $(3072)) % $(8)) * $(3072)) + (((r281_z / $(96)) % $(32)) * $(96)) + (r281_z % $(96))), dtype=tl.float32) - r281_dot_grad) * tl.load($(candidate_vsoftmax_5_ptr) + ((((r281_z / $(3072)) % $(8)) * $(3072)) + (((r281_z / $(96)) % $(32)) * $(96)) + (r281_z % $(96))), dtype=tl.float32)
    r281_value = r281_value.to(tl.float32)
    r281_value = tl.broadcast_to(r281_value, [$(24576)])
    tl.store($(candidate_vsoftmax_backward_data_3_ptr) + r281_z, r281_value)
    r282_z = tl.arange($(24576))
    r282_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r282_z / $(96)) % $(32)) * $(96)) + (r282_z % $(96))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_3_ptr) + ((((r282_z / $(3072)) % $(8)) * $(3072)) + (((r282_z / $(96)) % $(32)) * $(96)) + (r282_z % $(96))), dtype=tl.float32))
    r282_value = r282_value.to(tl.float32)
    r282_value = tl.broadcast_to(r282_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_12_ptr) + r282_z, r282_value)
    r283_z = tl.arange($(24576))
    r283_value = (tl.load($(candidate_vmasked_fill_12_ptr) + ((((r283_z / $(3072)) % $(8)) * $(3072)) + (((r283_z / $(96)) % $(32)) * $(96)) + (r283_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r283_value = r283_value.to(tl.float32)
    r283_value = tl.broadcast_to(r283_value, [$(24576)])
    tl.store($(candidate_vmul_33_ptr) + r283_z, r283_value)
    r284_row = tl.arange($(192))
    r284_col = tl.arange($(96))
    r284_lane = tl.arange($(32))
    for r284_group in range($(8)) {
      r284_left = tl.load($(candidate_vto_copy_37_ptr) + (((((((r284_group % $(8)) * $(6144)) + (r284_lane[None, :] * $(192)) + r284_row[:, None]) / $(6144)) % $(8)) * $(6144)) + ((((((r284_group % $(8)) * $(6144)) + (r284_lane[None, :] * $(192)) + r284_row[:, None]) / $(192)) % $(32)) * $(192)) + ((((r284_group % $(8)) * $(6144)) + (r284_lane[None, :] * $(192)) + r284_row[:, None]) % $(192))), dtype=tl.float32)
      r284_right = tl.load($(candidate_vmul_33_ptr) + (((((((r284_group % $(8)) * $(3072)) + (r284_lane[:, None] * $(96)) + r284_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r284_group % $(8)) * $(3072)) + (r284_lane[:, None] * $(96)) + r284_col[None, :]) / $(96)) % $(32)) * $(96)) + ((((r284_group % $(8)) * $(3072)) + (r284_lane[:, None] * $(96)) + r284_col[None, :]) % $(96))), dtype=tl.float32)
      r284_value = tl.dot(r284_left, r284_right)
      r284_value = r284_value.to(tl.float32)
      tl.store($(candidate_vbmm_56_ptr) + ((r284_group * $(18432)) + (r284_row[:, None] * $(96)) + r284_col[None, :]), r284_value)
    }
    r285_row = tl.arange($(32))
    r285_col = tl.arange($(192))
    r285_lane = tl.arange($(96))
    for r285_group in range($(8)) {
      r285_left = tl.load($(candidate_vmul_33_ptr) + (((((((r285_group % $(8)) * $(3072)) + (r285_row[:, None] * $(96)) + r285_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r285_group % $(8)) * $(3072)) + (r285_row[:, None] * $(96)) + r285_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r285_group % $(8)) * $(3072)) + (r285_row[:, None] * $(96)) + r285_lane[None, :]) % $(96))), dtype=tl.float32)
      r285_right = tl.load($(candidate_vto_copy_38_ptr) + (((((((r285_group % $(8)) * $(18432)) + (r285_col[None, :] * $(96)) + r285_lane[:, None]) / $(18432)) % $(8)) * $(18432)) + (((((r285_group % $(8)) * $(18432)) + (r285_col[None, :] * $(96)) + r285_lane[:, None]) % $(96)) * $(192)) + (((((r285_group % $(8)) * $(18432)) + (r285_col[None, :] * $(96)) + r285_lane[:, None]) / $(96)) % $(192))), dtype=tl.float32)
      r285_value = tl.dot(r285_left, r285_right)
      r285_value = r285_value.to(tl.float32)
      tl.store($(candidate_vbmm_57_ptr) + ((r285_group * $(6144)) + (r285_row[:, None] * $(192)) + r285_col[None, :]), r285_value)
    }
    r286_z = tl.arange($(98304))
    r286_value = tl.where(((((r286_z / $(1024)) % $(96)) >= $(0)) & (((r286_z / $(1024)) % $(96)) < $(96)) & ((((r286_z / $(1024)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_54_ptr) + ((((((((r286_z / $(128)) % $(8)) * $(12288)) + ((((r286_z / $(1024)) % $(96)) - $(0)) * $(128)) + (r286_z % $(128))) / $(12288)) % $(8)) * $(12288)) + ((((((r286_z / $(128)) % $(8)) * $(96)) + (((r286_z / $(1024)) % $(96)) - $(0))) % $(96)) * $(128)) + (r286_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r286_value = r286_value.to(tl.bfloat16)
    r286_value = tl.broadcast_to(r286_value, [$(98304)])
    tl.store($(candidate_vslice_backward_21_ptr) + r286_z, r286_value)
    r287_z = tl.arange($(147456))
    r287_value = tl.load($(candidate_vbmm_56_ptr) + ((((r287_z / $(18432)) % $(8)) * $(18432)) + ((r287_z % $(192)) * $(96)) + ((r287_z / $(192)) % $(96))), dtype=tl.float32)
    r287_value = r287_value.to(tl.bfloat16)
    r287_value = tl.broadcast_to(r287_value, [$(147456)])
    tl.store($(candidate_vto_copy_59_ptr) + r287_z, r287_value)
    r288_z = tl.arange($(147456))
    r288_value = tl.where(((((r288_z / $(1536)) % $(96)) >= $(0)) & (((r288_z / $(1536)) % $(96)) < $(96)) & ((((r288_z / $(1536)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_59_ptr) + ((((r288_z / $(192)) % $(8)) * $(18432)) + (((((r288_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(192)) + (r288_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r288_value = r288_value.to(tl.bfloat16)
    r288_value = tl.broadcast_to(r288_value, [$(147456)])
    tl.store($(candidate_vslice_backward_22_ptr) + r288_z, r288_value)
    r289_z = tl.arange($(49152))
    r289_value = tl.load($(candidate_vbmm_57_ptr) + ((((r289_z / $(6144)) % $(8)) * $(6144)) + (((r289_z / $(192)) % $(32)) * $(192)) + (r289_z % $(192))), dtype=tl.float32)
    r289_value = r289_value.to(tl.bfloat16)
    r289_value = tl.broadcast_to(r289_value, [$(49152)])
    tl.store($(candidate_vto_copy_60_ptr) + r289_z, r289_value)
    r290_z = tl.arange($(49152))
    r290_value = tl.where(((((r290_z / $(1536)) % $(32)) >= $(0)) & (((r290_z / $(1536)) % $(32)) < $(32)) & ((((r290_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_60_ptr) + ((((r290_z / $(192)) % $(8)) * $(6144)) + (((((r290_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(192)) + (r290_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r290_value = r290_value.to(tl.bfloat16)
    r290_value = tl.broadcast_to(r290_value, [$(49152)])
    tl.store($(candidate_vslice_backward_23_ptr) + r290_z, r290_value)
    r291_z = tl.arange($(0))
    r291_value = tl.cast(0.0, tl.bfloat16)
    r291_value = r291_value.to(tl.bfloat16)
    r291_value = tl.broadcast_to(r291_value, [$(0)])
    tl.store($(candidate_vfill_6_ptr) + r291_z, r291_value)
    r292_z = tl.arange($(98304))
    r292_value = tl.where(((((r292_z / $(1024)) % $(96)) >= $(96)) & (((r292_z / $(1024)) % $(96)) < $(96)) & ((((r292_z / $(1024)) % $(96)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_6_ptr) + ((((((r292_z / $(1024)) % $(96)) - $(96)) % $(0)) * $(1024)) + (((r292_z / $(128)) % $(8)) * $(128)) + (r292_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_21_ptr) + ((((r292_z / $(1024)) % $(96)) * $(1024)) + (((r292_z / $(128)) % $(8)) * $(128)) + (r292_z % $(128))), dtype=tl.bfloat16))
    r292_value = r292_value.to(tl.bfloat16)
    r292_value = tl.broadcast_to(r292_value, [$(98304)])
    tl.store($(candidate_vslice_scatter_6_ptr) + r292_z, r292_value)
    r293_z = tl.arange($(0))
    r293_value = tl.cast(0.0, tl.bfloat16)
    r293_value = r293_value.to(tl.bfloat16)
    r293_value = tl.broadcast_to(r293_value, [$(0)])
    tl.store($(candidate_vfill_7_ptr) + r293_z, r293_value)
    r294_z = tl.arange($(147456))
    r294_value = tl.where(((((r294_z / $(1536)) % $(96)) >= $(96)) & (((r294_z / $(1536)) % $(96)) < $(96)) & ((((r294_z / $(1536)) % $(96)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_7_ptr) + ((((((r294_z / $(1536)) % $(96)) - $(96)) % $(0)) * $(1536)) + (((r294_z / $(192)) % $(8)) * $(192)) + (r294_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_22_ptr) + ((((r294_z / $(1536)) % $(96)) * $(1536)) + (((r294_z / $(192)) % $(8)) * $(192)) + (r294_z % $(192))), dtype=tl.bfloat16))
    r294_value = r294_value.to(tl.bfloat16)
    r294_value = tl.broadcast_to(r294_value, [$(147456)])
    tl.store($(candidate_vslice_scatter_7_ptr) + r294_z, r294_value)
    r295_z = tl.arange($(6144))
    r295_lane = tl.arange($(8))
    r295_value = tl.sum(tl.load($(candidate_vslice_scatter_7_ptr) + ((((r295_z[:, None] / $(64)) % $(96)) * $(1536)) + (r295_lane[None, :] * $(192)) + (r295_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r295_value = r295_value.to(tl.bfloat16)
    r295_value = tl.broadcast_to(r295_value, [$(6144)])
    tl.store($(candidate_vsum_4_ptr) + r295_z, r295_value)
    r296_row = tl.arange($(96))
    r296_col = tl.arange($(512))
    r296_lane = tl.arange($(1024))
    for r296_group in range($(1)) {
      r296_left = tl.load($(candidate_vslice_scatter_6_ptr) + ((((((r296_row[:, None] * $(1024)) + r296_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + (((((r296_row[:, None] * $(1024)) + r296_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r296_row[:, None] * $(1024)) + r296_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r296_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r296_col[None, :] * $(1024)) + r296_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r296_col[None, :] * $(1024)) + r296_lane[:, None]) % $(128)) * $(512)) + ((((r296_col[None, :] * $(1024)) + r296_lane[:, None]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r296_value = tl.dot(r296_left, r296_right)
      r296_value = r296_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_58_ptr) + ((r296_group * $(49152)) + (r296_row[:, None] * $(512)) + r296_col[None, :]), r296_value)
    }
    r297_row = tl.arange($(96))
    r297_col = tl.arange($(512))
    r297_lane = tl.arange($(1024))
    for r297_group in range($(1)) {
      r297_left = tl.load($(candidate_vslice_scatter_7_ptr) + ((((((r297_row[:, None] * $(1024)) + r297_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + (((((r297_row[:, None] * $(1024)) + r297_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r297_row[:, None] * $(1024)) + r297_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r297_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r297_col[None, :] * $(1024)) + r297_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r297_col[None, :] * $(1024)) + r297_lane[:, None]) % $(128)) * $(512)) + ((((r297_col[None, :] * $(1024)) + r297_lane[:, None]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r297_value = tl.dot(r297_left, r297_right)
      r297_value = r297_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_59_ptr) + ((r297_group * $(49152)) + (r297_row[:, None] * $(512)) + r297_col[None, :]), r297_value)
    }
    r298_z = tl.arange($(49152))
    r298_value = (tl.load($(candidate_vbmm_58_ptr) + ((((r298_z / $(512)) % $(96)) * $(512)) + (r298_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_59_ptr) + ((((r298_z / $(512)) % $(96)) * $(512)) + (r298_z % $(512))), dtype=tl.bfloat16))
    r298_value = r298_value.to(tl.bfloat16)
    r298_value = tl.broadcast_to(r298_value, [$(49152)])
    tl.store($(candidate_vadd_26_ptr) + r298_z, r298_value)
    r299_z = tl.arange($(49152))
    r299_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_26_ptr) + ((((r299_z / $(512)) % $(96)) * $(512)) + (r299_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r299_value = r299_value.to(tl.bfloat16)
    r299_value = tl.broadcast_to(r299_value, [$(49152)])
    tl.store($(candidate_vselect_backward_3_ptr) + r299_z, r299_value)
    r300_z = tl.arange($(147456))
    r300_value = tl.where(((((r300_z / $(512)) % $(288)) >= $(96)) & (((r300_z / $(512)) % $(288)) < $(192)) & ((((r300_z / $(512)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_3_ptr) + ((((((r300_z / $(512)) % $(288)) - $(96)) % $(96)) * $(512)) + (r300_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r300_value = r300_value.to(tl.bfloat16)
    r300_value = tl.broadcast_to(r300_value, [$(147456)])
    tl.store($(candidate_vslice_backward_24_ptr) + r300_z, r300_value)
    r301_z = tl.arange($(147456))
    r301_value = (tl.load($(candidate_vadd_22_ptr) + ((((r301_z / $(512)) % $(288)) * $(512)) + (r301_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_24_ptr) + ((((r301_z / $(512)) % $(288)) * $(512)) + (r301_z % $(512))), dtype=tl.bfloat16))
    r301_value = r301_value.to(tl.bfloat16)
    r301_value = tl.broadcast_to(r301_value, [$(147456)])
    tl.store($(candidate_vadd_27_ptr) + r301_z, r301_value)
    r302_z = tl.arange($(18432))
    r302_value = tl.where(((((r302_z / $(64)) % $(288)) >= $(96)) & (((r302_z / $(64)) % $(288)) < $(192)) & ((((r302_z / $(64)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vsum_4_ptr) + ((((((r302_z / $(64)) % $(288)) - $(96)) % $(96)) * $(64)) + (r302_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r302_value = r302_value.to(tl.bfloat16)
    r302_value = tl.broadcast_to(r302_value, [$(18432)])
    tl.store($(candidate_vslice_backward_25_ptr) + r302_z, r302_value)
    r303_z = tl.arange($(18432))
    r303_value = (tl.load($(candidate_vadd_23_ptr) + ((((r303_z / $(64)) % $(288)) * $(64)) + (r303_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_25_ptr) + ((((r303_z / $(64)) % $(288)) * $(64)) + (r303_z % $(64))), dtype=tl.bfloat16))
    r303_value = r303_value.to(tl.bfloat16)
    r303_value = tl.broadcast_to(r303_value, [$(18432)])
    tl.store($(candidate_vadd_28_ptr) + r303_z, r303_value)
    r304_z = tl.arange($(294912))
    r304_value = tl.where(((((r304_z / $(1024)) % $(288)) >= $(160)) & (((r304_z / $(1024)) % $(288)) < $(192)) & ((((r304_z / $(1024)) % $(288)) - $(160)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_23_ptr) + ((((((r304_z / $(1024)) % $(288)) - $(160)) % $(32)) * $(1536)) + (((r304_z / $(128)) % $(8)) * $(192)) + (r304_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r304_value = r304_value.to(tl.bfloat16)
    r304_value = tl.broadcast_to(r304_value, [$(294912)])
    tl.store($(candidate_vslice_backward_26_ptr) + r304_z, r304_value)
    r305_z = tl.arange($(294912))
    r305_value = (tl.load($(candidate_vadd_24_ptr) + ((((r305_z / $(1024)) % $(288)) * $(1024)) + (((r305_z / $(128)) % $(8)) * $(128)) + (r305_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_26_ptr) + ((((r305_z / $(1024)) % $(288)) * $(1024)) + (((r305_z / $(128)) % $(8)) * $(128)) + (r305_z % $(128))), dtype=tl.bfloat16))
    r305_value = r305_value.to(tl.bfloat16)
    r305_value = tl.broadcast_to(r305_value, [$(294912)])
    tl.store($(candidate_vadd_29_ptr) + r305_z, r305_value)
    r306_z = tl.arange($(147456))
    r306_value = tl.where(((((r306_z / $(512)) % $(288)) >= $(160)) & (((r306_z / $(512)) % $(288)) < $(192)) & ((((r306_z / $(512)) % $(288)) - $(160)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_23_ptr) + ((((((r306_z / $(512)) % $(288)) - $(160)) % $(32)) * $(1536)) + (((r306_z / $(64)) % $(8)) * $(192)) + (r306_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r306_value = r306_value.to(tl.bfloat16)
    r306_value = tl.broadcast_to(r306_value, [$(147456)])
    tl.store($(candidate_vslice_backward_27_ptr) + r306_z, r306_value)
    r307_z = tl.arange($(147456))
    r307_value = (tl.load($(candidate_vadd_25_ptr) + ((((r307_z / $(512)) % $(288)) * $(512)) + (((r307_z / $(64)) % $(8)) * $(64)) + (r307_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_27_ptr) + ((((r307_z / $(512)) % $(288)) * $(512)) + (((r307_z / $(64)) % $(8)) * $(64)) + (r307_z % $(64))), dtype=tl.bfloat16))
    r307_value = r307_value.to(tl.bfloat16)
    r307_value = tl.broadcast_to(r307_value, [$(147456)])
    tl.store($(candidate_vadd_30_ptr) + r307_z, r307_value)
    r308_row = tl.arange($(64))
    r308_col = tl.arange($(128))
    r308_lane = tl.arange($(32))
    for r308_group in range($(8)) {
      r308_left = tl.load($(candidate_vto_copy_36_ptr) + (((((((r308_group % $(8)) * $(2048)) + (r308_lane[None, :] * $(64)) + r308_row[:, None]) / $(2048)) % $(8)) * $(2048)) + ((((((r308_group % $(8)) * $(2048)) + (r308_lane[None, :] * $(64)) + r308_row[:, None]) / $(64)) % $(32)) * $(64)) + ((((r308_group % $(8)) * $(2048)) + (r308_lane[None, :] * $(64)) + r308_row[:, None]) % $(64))), dtype=tl.bfloat16)
      r308_right = tl.load($(candidate_vmm_4_ptr) + (((((((r308_group % $(8)) * $(4096)) + (r308_lane[:, None] * $(128)) + r308_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r308_group % $(8)) * $(4096)) + (r308_lane[:, None] * $(128)) + r308_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r308_group % $(8)) * $(4096)) + (r308_lane[:, None] * $(128)) + r308_col[None, :]) % $(128)) + $(131072)), dtype=tl.bfloat16)
      r308_value = tl.dot(r308_left, r308_right)
      r308_value = r308_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_60_ptr) + ((r308_group * $(8192)) + (r308_row[:, None] * $(128)) + r308_col[None, :]), r308_value)
    }
    r309_row = tl.arange($(32))
    r309_col = tl.arange($(64))
    r309_lane = tl.arange($(128))
    for r309_group in range($(8)) {
      r309_left = tl.load($(candidate_vmm_4_ptr) + (((((((r309_group % $(8)) * $(4096)) + (r309_row[:, None] * $(128)) + r309_lane[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r309_group % $(8)) * $(4096)) + (r309_row[:, None] * $(128)) + r309_lane[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r309_group % $(8)) * $(4096)) + (r309_row[:, None] * $(128)) + r309_lane[None, :]) % $(128)) + $(131072)), dtype=tl.bfloat16)
      r309_right = tl.load($(candidate_vbmm_17_ptr) + (((((((r309_group % $(8)) * $(8192)) + (r309_col[None, :] * $(128)) + r309_lane[:, None]) / $(128)) % $(64)) * $(1024)) + ((((((r309_group % $(8)) * $(8192)) + (r309_col[None, :] * $(128)) + r309_lane[:, None]) / $(8192)) % $(8)) * $(128)) + ((((r309_group % $(8)) * $(8192)) + (r309_col[None, :] * $(128)) + r309_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r309_value = tl.dot(r309_left, r309_right)
      r309_value = r309_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_61_ptr) + ((r309_group * $(2048)) + (r309_row[:, None] * $(64)) + r309_col[None, :]), r309_value)
    }
    r310_z = tl.arange($(16384))
    r310_value = tl.load($(candidate_vbmm_61_ptr) + ((((r310_z / $(2048)) % $(8)) * $(2048)) + (((r310_z / $(64)) % $(32)) * $(64)) + (r310_z % $(64))), dtype=tl.bfloat16)
    r310_value = r310_value.to(tl.float32)
    r310_value = tl.broadcast_to(r310_value, [$(16384)])
    tl.store($(candidate_vto_copy_61_ptr) + r310_z, r310_value)
    r311_z = tl.arange($(16384))
    r311_lane = tl.arange($(64))
    r311_row_values = tl.load($(candidate_vsoftmax_4_ptr) + ((((r311_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r311_z[:, None] / $(64)) % $(32)) * $(64)) + (r311_lane[None, :] % $(64))), dtype=tl.float32)
    r311_row_grad = tl.load($(candidate_vto_copy_61_ptr) + ((((r311_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r311_z[:, None] / $(64)) % $(32)) * $(64)) + (r311_lane[None, :] % $(64))), dtype=tl.float32)
    r311_weighted = r311_row_grad * r311_row_values
    r311_dot_grad = tl.sum(r311_weighted, axis=1)
    r311_value = (tl.load($(candidate_vto_copy_61_ptr) + ((((r311_z / $(2048)) % $(8)) * $(2048)) + (((r311_z / $(64)) % $(32)) * $(64)) + (r311_z % $(64))), dtype=tl.float32) - r311_dot_grad) * tl.load($(candidate_vsoftmax_4_ptr) + ((((r311_z / $(2048)) % $(8)) * $(2048)) + (((r311_z / $(64)) % $(32)) * $(64)) + (r311_z % $(64))), dtype=tl.float32)
    r311_value = r311_value.to(tl.float32)
    r311_value = tl.broadcast_to(r311_value, [$(16384)])
    tl.store($(candidate_vsoftmax_backward_data_4_ptr) + r311_z, r311_value)
    r312_z = tl.arange($(16384))
    r312_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r312_z / $(64)) % $(32)) * $(64)) + (r312_z % $(64))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_4_ptr) + ((((r312_z / $(2048)) % $(8)) * $(2048)) + (((r312_z / $(64)) % $(32)) * $(64)) + (r312_z % $(64))), dtype=tl.float32))
    r312_value = r312_value.to(tl.float32)
    r312_value = tl.broadcast_to(r312_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_13_ptr) + r312_z, r312_value)
    r313_z = tl.arange($(16384))
    r313_value = (tl.load($(candidate_vmasked_fill_13_ptr) + ((((r313_z / $(2048)) % $(8)) * $(2048)) + (((r313_z / $(64)) % $(32)) * $(64)) + (r313_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r313_value = r313_value.to(tl.float32)
    r313_value = tl.broadcast_to(r313_value, [$(16384)])
    tl.store($(candidate_vmul_34_ptr) + r313_z, r313_value)
    r314_row = tl.arange($(192))
    r314_col = tl.arange($(64))
    r314_lane = tl.arange($(32))
    for r314_group in range($(8)) {
      r314_left = tl.load($(candidate_vto_copy_34_ptr) + (((((((r314_group % $(8)) * $(6144)) + (r314_lane[None, :] * $(192)) + r314_row[:, None]) / $(6144)) % $(8)) * $(6144)) + ((((((r314_group % $(8)) * $(6144)) + (r314_lane[None, :] * $(192)) + r314_row[:, None]) / $(192)) % $(32)) * $(192)) + ((((r314_group % $(8)) * $(6144)) + (r314_lane[None, :] * $(192)) + r314_row[:, None]) % $(192))), dtype=tl.float32)
      r314_right = tl.load($(candidate_vmul_34_ptr) + (((((((r314_group % $(8)) * $(2048)) + (r314_lane[:, None] * $(64)) + r314_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r314_group % $(8)) * $(2048)) + (r314_lane[:, None] * $(64)) + r314_col[None, :]) / $(64)) % $(32)) * $(64)) + ((((r314_group % $(8)) * $(2048)) + (r314_lane[:, None] * $(64)) + r314_col[None, :]) % $(64))), dtype=tl.float32)
      r314_value = tl.dot(r314_left, r314_right)
      r314_value = r314_value.to(tl.float32)
      tl.store($(candidate_vbmm_62_ptr) + ((r314_group * $(12288)) + (r314_row[:, None] * $(64)) + r314_col[None, :]), r314_value)
    }
    r315_row = tl.arange($(32))
    r315_col = tl.arange($(192))
    r315_lane = tl.arange($(64))
    for r315_group in range($(8)) {
      r315_left = tl.load($(candidate_vmul_34_ptr) + (((((((r315_group % $(8)) * $(2048)) + (r315_row[:, None] * $(64)) + r315_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r315_group % $(8)) * $(2048)) + (r315_row[:, None] * $(64)) + r315_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r315_group % $(8)) * $(2048)) + (r315_row[:, None] * $(64)) + r315_lane[None, :]) % $(64))), dtype=tl.float32)
      r315_right = tl.load($(candidate_vto_copy_35_ptr) + (((((((r315_group % $(8)) * $(12288)) + (r315_col[None, :] * $(64)) + r315_lane[:, None]) / $(12288)) % $(8)) * $(12288)) + (((((r315_group % $(8)) * $(12288)) + (r315_col[None, :] * $(64)) + r315_lane[:, None]) % $(64)) * $(192)) + (((((r315_group % $(8)) * $(12288)) + (r315_col[None, :] * $(64)) + r315_lane[:, None]) / $(64)) % $(192))), dtype=tl.float32)
      r315_value = tl.dot(r315_left, r315_right)
      r315_value = r315_value.to(tl.float32)
      tl.store($(candidate_vbmm_63_ptr) + ((r315_group * $(6144)) + (r315_row[:, None] * $(192)) + r315_col[None, :]), r315_value)
    }
    r316_z = tl.arange($(98304))
    r316_value = tl.where(((((r316_z / $(1024)) % $(96)) >= $(0)) & (((r316_z / $(1024)) % $(96)) < $(64)) & ((((r316_z / $(1024)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_60_ptr) + ((((((((r316_z / $(128)) % $(8)) * $(8192)) + ((((r316_z / $(1024)) % $(96)) - $(0)) * $(128)) + (r316_z % $(128))) / $(8192)) % $(8)) * $(8192)) + ((((((r316_z / $(128)) % $(8)) * $(64)) + (((r316_z / $(1024)) % $(96)) - $(0))) % $(64)) * $(128)) + (r316_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r316_value = r316_value.to(tl.bfloat16)
    r316_value = tl.broadcast_to(r316_value, [$(98304)])
    tl.store($(candidate_vslice_backward_28_ptr) + r316_z, r316_value)
    r317_z = tl.arange($(98304))
    r317_value = tl.load($(candidate_vbmm_62_ptr) + ((((r317_z / $(12288)) % $(8)) * $(12288)) + ((r317_z % $(192)) * $(64)) + ((r317_z / $(192)) % $(64))), dtype=tl.float32)
    r317_value = r317_value.to(tl.bfloat16)
    r317_value = tl.broadcast_to(r317_value, [$(98304)])
    tl.store($(candidate_vto_copy_62_ptr) + r317_z, r317_value)
    r318_z = tl.arange($(147456))
    r318_value = tl.where(((((r318_z / $(1536)) % $(96)) >= $(0)) & (((r318_z / $(1536)) % $(96)) < $(64)) & ((((r318_z / $(1536)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_62_ptr) + ((((r318_z / $(192)) % $(8)) * $(12288)) + (((((r318_z / $(1536)) % $(96)) - $(0)) % $(64)) * $(192)) + (r318_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r318_value = r318_value.to(tl.bfloat16)
    r318_value = tl.broadcast_to(r318_value, [$(147456)])
    tl.store($(candidate_vslice_backward_29_ptr) + r318_z, r318_value)
    r319_z = tl.arange($(49152))
    r319_value = tl.load($(candidate_vbmm_63_ptr) + ((((r319_z / $(6144)) % $(8)) * $(6144)) + (((r319_z / $(192)) % $(32)) * $(192)) + (r319_z % $(192))), dtype=tl.float32)
    r319_value = r319_value.to(tl.bfloat16)
    r319_value = tl.broadcast_to(r319_value, [$(49152)])
    tl.store($(candidate_vto_copy_63_ptr) + r319_z, r319_value)
    r320_z = tl.arange($(49152))
    r320_value = tl.where(((((r320_z / $(1536)) % $(32)) >= $(0)) & (((r320_z / $(1536)) % $(32)) < $(32)) & ((((r320_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_63_ptr) + ((((r320_z / $(192)) % $(8)) * $(6144)) + (((((r320_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(192)) + (r320_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r320_value = r320_value.to(tl.bfloat16)
    r320_value = tl.broadcast_to(r320_value, [$(49152)])
    tl.store($(candidate_vslice_backward_30_ptr) + r320_z, r320_value)
    r321_z = tl.arange($(32768))
    r321_value = tl.cast(0.0, tl.bfloat16)
    r321_value = r321_value.to(tl.bfloat16)
    r321_value = tl.broadcast_to(r321_value, [$(32768)])
    tl.store($(candidate_vfill_8_ptr) + r321_z, r321_value)
    r322_z = tl.arange($(98304))
    r322_value = tl.where(((((r322_z / $(1024)) % $(96)) >= $(64)) & (((r322_z / $(1024)) % $(96)) < $(96)) & ((((r322_z / $(1024)) % $(96)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_8_ptr) + ((((((r322_z / $(1024)) % $(96)) - $(64)) % $(32)) * $(1024)) + (((r322_z / $(128)) % $(8)) * $(128)) + (r322_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_28_ptr) + ((((r322_z / $(1024)) % $(96)) * $(1024)) + (((r322_z / $(128)) % $(8)) * $(128)) + (r322_z % $(128))), dtype=tl.bfloat16))
    r322_value = r322_value.to(tl.bfloat16)
    r322_value = tl.broadcast_to(r322_value, [$(98304)])
    tl.store($(candidate_vslice_scatter_8_ptr) + r322_z, r322_value)
    r323_z = tl.arange($(49152))
    r323_value = tl.cast(0.0, tl.bfloat16)
    r323_value = r323_value.to(tl.bfloat16)
    r323_value = tl.broadcast_to(r323_value, [$(49152)])
    tl.store($(candidate_vfill_9_ptr) + r323_z, r323_value)
    r324_z = tl.arange($(147456))
    r324_value = tl.where(((((r324_z / $(1536)) % $(96)) >= $(64)) & (((r324_z / $(1536)) % $(96)) < $(96)) & ((((r324_z / $(1536)) % $(96)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_9_ptr) + ((((((r324_z / $(1536)) % $(96)) - $(64)) % $(32)) * $(1536)) + (((r324_z / $(192)) % $(8)) * $(192)) + (r324_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_29_ptr) + ((((r324_z / $(1536)) % $(96)) * $(1536)) + (((r324_z / $(192)) % $(8)) * $(192)) + (r324_z % $(192))), dtype=tl.bfloat16))
    r324_value = r324_value.to(tl.bfloat16)
    r324_value = tl.broadcast_to(r324_value, [$(147456)])
    tl.store($(candidate_vslice_scatter_9_ptr) + r324_z, r324_value)
    r325_z = tl.arange($(6144))
    r325_lane = tl.arange($(8))
    r325_value = tl.sum(tl.load($(candidate_vslice_scatter_9_ptr) + ((((r325_z[:, None] / $(64)) % $(96)) * $(1536)) + (r325_lane[None, :] * $(192)) + (r325_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r325_value = r325_value.to(tl.bfloat16)
    r325_value = tl.broadcast_to(r325_value, [$(6144)])
    tl.store($(candidate_vsum_5_ptr) + r325_z, r325_value)
    r326_row = tl.arange($(96))
    r326_col = tl.arange($(512))
    r326_lane = tl.arange($(1024))
    for r326_group in range($(1)) {
      r326_left = tl.load($(candidate_vslice_scatter_8_ptr) + ((((((r326_row[:, None] * $(1024)) + r326_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + (((((r326_row[:, None] * $(1024)) + r326_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r326_row[:, None] * $(1024)) + r326_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r326_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r326_col[None, :] * $(1024)) + r326_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r326_col[None, :] * $(1024)) + r326_lane[:, None]) % $(128)) * $(512)) + ((((r326_col[None, :] * $(1024)) + r326_lane[:, None]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r326_value = tl.dot(r326_left, r326_right)
      r326_value = r326_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_64_ptr) + ((r326_group * $(49152)) + (r326_row[:, None] * $(512)) + r326_col[None, :]), r326_value)
    }
    r327_row = tl.arange($(96))
    r327_col = tl.arange($(512))
    r327_lane = tl.arange($(1024))
    for r327_group in range($(1)) {
      r327_left = tl.load($(candidate_vslice_scatter_9_ptr) + ((((((r327_row[:, None] * $(1024)) + r327_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + (((((r327_row[:, None] * $(1024)) + r327_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r327_row[:, None] * $(1024)) + r327_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r327_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r327_col[None, :] * $(1024)) + r327_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r327_col[None, :] * $(1024)) + r327_lane[:, None]) % $(128)) * $(512)) + ((((r327_col[None, :] * $(1024)) + r327_lane[:, None]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r327_value = tl.dot(r327_left, r327_right)
      r327_value = r327_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_65_ptr) + ((r327_group * $(49152)) + (r327_row[:, None] * $(512)) + r327_col[None, :]), r327_value)
    }
    r328_z = tl.arange($(49152))
    r328_value = (tl.load($(candidate_vbmm_64_ptr) + ((((r328_z / $(512)) % $(96)) * $(512)) + (r328_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_65_ptr) + ((((r328_z / $(512)) % $(96)) * $(512)) + (r328_z % $(512))), dtype=tl.bfloat16))
    r328_value = r328_value.to(tl.bfloat16)
    r328_value = tl.broadcast_to(r328_value, [$(49152)])
    tl.store($(candidate_vadd_31_ptr) + r328_z, r328_value)
    r329_z = tl.arange($(49152))
    r329_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_31_ptr) + ((((r329_z / $(512)) % $(96)) * $(512)) + (r329_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r329_value = r329_value.to(tl.bfloat16)
    r329_value = tl.broadcast_to(r329_value, [$(49152)])
    tl.store($(candidate_vselect_backward_4_ptr) + r329_z, r329_value)
    r330_z = tl.arange($(147456))
    r330_value = tl.where(((((r330_z / $(512)) % $(288)) >= $(96)) & (((r330_z / $(512)) % $(288)) < $(192)) & ((((r330_z / $(512)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_4_ptr) + ((((((r330_z / $(512)) % $(288)) - $(96)) % $(96)) * $(512)) + (r330_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r330_value = r330_value.to(tl.bfloat16)
    r330_value = tl.broadcast_to(r330_value, [$(147456)])
    tl.store($(candidate_vslice_backward_31_ptr) + r330_z, r330_value)
    r331_z = tl.arange($(147456))
    r331_value = (tl.load($(candidate_vadd_27_ptr) + ((((r331_z / $(512)) % $(288)) * $(512)) + (r331_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_31_ptr) + ((((r331_z / $(512)) % $(288)) * $(512)) + (r331_z % $(512))), dtype=tl.bfloat16))
    r331_value = r331_value.to(tl.bfloat16)
    r331_value = tl.broadcast_to(r331_value, [$(147456)])
    tl.store($(candidate_vadd_32_ptr) + r331_z, r331_value)
    r332_z = tl.arange($(18432))
    r332_value = tl.where(((((r332_z / $(64)) % $(288)) >= $(96)) & (((r332_z / $(64)) % $(288)) < $(192)) & ((((r332_z / $(64)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vsum_5_ptr) + ((((((r332_z / $(64)) % $(288)) - $(96)) % $(96)) * $(64)) + (r332_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r332_value = r332_value.to(tl.bfloat16)
    r332_value = tl.broadcast_to(r332_value, [$(18432)])
    tl.store($(candidate_vslice_backward_32_ptr) + r332_z, r332_value)
    r333_z = tl.arange($(18432))
    r333_value = (tl.load($(candidate_vadd_28_ptr) + ((((r333_z / $(64)) % $(288)) * $(64)) + (r333_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_32_ptr) + ((((r333_z / $(64)) % $(288)) * $(64)) + (r333_z % $(64))), dtype=tl.bfloat16))
    r333_value = r333_value.to(tl.bfloat16)
    r333_value = tl.broadcast_to(r333_value, [$(18432)])
    tl.store($(candidate_vadd_33_ptr) + r333_z, r333_value)
    r334_z = tl.arange($(294912))
    r334_value = tl.where(((((r334_z / $(1024)) % $(288)) >= $(128)) & (((r334_z / $(1024)) % $(288)) < $(160)) & ((((r334_z / $(1024)) % $(288)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_30_ptr) + ((((((r334_z / $(1024)) % $(288)) - $(128)) % $(32)) * $(1536)) + (((r334_z / $(128)) % $(8)) * $(192)) + (r334_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r334_value = r334_value.to(tl.bfloat16)
    r334_value = tl.broadcast_to(r334_value, [$(294912)])
    tl.store($(candidate_vslice_backward_33_ptr) + r334_z, r334_value)
    r335_z = tl.arange($(294912))
    r335_value = (tl.load($(candidate_vadd_29_ptr) + ((((r335_z / $(1024)) % $(288)) * $(1024)) + (((r335_z / $(128)) % $(8)) * $(128)) + (r335_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_33_ptr) + ((((r335_z / $(1024)) % $(288)) * $(1024)) + (((r335_z / $(128)) % $(8)) * $(128)) + (r335_z % $(128))), dtype=tl.bfloat16))
    r335_value = r335_value.to(tl.bfloat16)
    r335_value = tl.broadcast_to(r335_value, [$(294912)])
    tl.store($(candidate_vadd_34_ptr) + r335_z, r335_value)
    r336_z = tl.arange($(147456))
    r336_value = tl.where(((((r336_z / $(512)) % $(288)) >= $(128)) & (((r336_z / $(512)) % $(288)) < $(160)) & ((((r336_z / $(512)) % $(288)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_30_ptr) + ((((((r336_z / $(512)) % $(288)) - $(128)) % $(32)) * $(1536)) + (((r336_z / $(64)) % $(8)) * $(192)) + (r336_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r336_value = r336_value.to(tl.bfloat16)
    r336_value = tl.broadcast_to(r336_value, [$(147456)])
    tl.store($(candidate_vslice_backward_34_ptr) + r336_z, r336_value)
    r337_z = tl.arange($(147456))
    r337_value = (tl.load($(candidate_vadd_30_ptr) + ((((r337_z / $(512)) % $(288)) * $(512)) + (((r337_z / $(64)) % $(8)) * $(64)) + (r337_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_34_ptr) + ((((r337_z / $(512)) % $(288)) * $(512)) + (((r337_z / $(64)) % $(8)) * $(64)) + (r337_z % $(64))), dtype=tl.bfloat16))
    r337_value = r337_value.to(tl.bfloat16)
    r337_value = tl.broadcast_to(r337_value, [$(147456)])
    tl.store($(candidate_vadd_35_ptr) + r337_z, r337_value)
    r338_row = tl.arange($(32))
    r338_col = tl.arange($(128))
    r338_lane = tl.arange($(32))
    for r338_group in range($(8)) {
      r338_left = tl.load($(candidate_vto_copy_33_ptr) + (((((((r338_group % $(8)) * $(1024)) + (r338_lane[None, :] * $(32)) + r338_row[:, None]) / $(1024)) % $(8)) * $(1024)) + ((((((r338_group % $(8)) * $(1024)) + (r338_lane[None, :] * $(32)) + r338_row[:, None]) / $(32)) % $(32)) * $(32)) + ((((r338_group % $(8)) * $(1024)) + (r338_lane[None, :] * $(32)) + r338_row[:, None]) % $(32))), dtype=tl.bfloat16)
      r338_right = tl.load($(candidate_vmm_4_ptr) + (((((((r338_group % $(8)) * $(4096)) + (r338_lane[:, None] * $(128)) + r338_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r338_group % $(8)) * $(4096)) + (r338_lane[:, None] * $(128)) + r338_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r338_group % $(8)) * $(4096)) + (r338_lane[:, None] * $(128)) + r338_col[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r338_value = tl.dot(r338_left, r338_right)
      r338_value = r338_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_66_ptr) + ((r338_group * $(4096)) + (r338_row[:, None] * $(128)) + r338_col[None, :]), r338_value)
    }
    r339_row = tl.arange($(32))
    r339_col = tl.arange($(32))
    r339_lane = tl.arange($(128))
    for r339_group in range($(8)) {
      r339_left = tl.load($(candidate_vmm_4_ptr) + (((((((r339_group % $(8)) * $(4096)) + (r339_row[:, None] * $(128)) + r339_lane[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r339_group % $(8)) * $(4096)) + (r339_row[:, None] * $(128)) + r339_lane[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r339_group % $(8)) * $(4096)) + (r339_row[:, None] * $(128)) + r339_lane[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r339_right = tl.load($(candidate_vbmm_13_ptr) + (((((((r339_group % $(8)) * $(4096)) + (r339_col[None, :] * $(128)) + r339_lane[:, None]) / $(128)) % $(32)) * $(1024)) + ((((((r339_group % $(8)) * $(4096)) + (r339_col[None, :] * $(128)) + r339_lane[:, None]) / $(4096)) % $(8)) * $(128)) + ((((r339_group % $(8)) * $(4096)) + (r339_col[None, :] * $(128)) + r339_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r339_value = tl.dot(r339_left, r339_right)
      r339_value = r339_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_67_ptr) + ((r339_group * $(1024)) + (r339_row[:, None] * $(32)) + r339_col[None, :]), r339_value)
    }
    r340_z = tl.arange($(8192))
    r340_value = tl.load($(candidate_vbmm_67_ptr) + ((((r340_z / $(1024)) % $(8)) * $(1024)) + (((r340_z / $(32)) % $(32)) * $(32)) + (r340_z % $(32))), dtype=tl.bfloat16)
    r340_value = r340_value.to(tl.float32)
    r340_value = tl.broadcast_to(r340_value, [$(8192)])
    tl.store($(candidate_vto_copy_64_ptr) + r340_z, r340_value)
    r341_z = tl.arange($(8192))
    r341_lane = tl.arange($(32))
    r341_row_values = tl.load($(candidate_vsoftmax_3_ptr) + ((((r341_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r341_z[:, None] / $(32)) % $(32)) * $(32)) + (r341_lane[None, :] % $(32))), dtype=tl.float32)
    r341_row_grad = tl.load($(candidate_vto_copy_64_ptr) + ((((r341_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r341_z[:, None] / $(32)) % $(32)) * $(32)) + (r341_lane[None, :] % $(32))), dtype=tl.float32)
    r341_weighted = r341_row_grad * r341_row_values
    r341_dot_grad = tl.sum(r341_weighted, axis=1)
    r341_value = (tl.load($(candidate_vto_copy_64_ptr) + ((((r341_z / $(1024)) % $(8)) * $(1024)) + (((r341_z / $(32)) % $(32)) * $(32)) + (r341_z % $(32))), dtype=tl.float32) - r341_dot_grad) * tl.load($(candidate_vsoftmax_3_ptr) + ((((r341_z / $(1024)) % $(8)) * $(1024)) + (((r341_z / $(32)) % $(32)) * $(32)) + (r341_z % $(32))), dtype=tl.float32)
    r341_value = r341_value.to(tl.float32)
    r341_value = tl.broadcast_to(r341_value, [$(8192)])
    tl.store($(candidate_vsoftmax_backward_data_5_ptr) + r341_z, r341_value)
    r342_z = tl.arange($(8192))
    r342_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r342_z / $(32)) % $(32)) * $(32)) + (r342_z % $(32))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_5_ptr) + ((((r342_z / $(1024)) % $(8)) * $(1024)) + (((r342_z / $(32)) % $(32)) * $(32)) + (r342_z % $(32))), dtype=tl.float32))
    r342_value = r342_value.to(tl.float32)
    r342_value = tl.broadcast_to(r342_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_14_ptr) + r342_z, r342_value)
    r343_z = tl.arange($(8192))
    r343_value = (tl.load($(candidate_vmasked_fill_14_ptr) + ((((r343_z / $(1024)) % $(8)) * $(1024)) + (((r343_z / $(32)) % $(32)) * $(32)) + (r343_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r343_value = r343_value.to(tl.float32)
    r343_value = tl.broadcast_to(r343_value, [$(8192)])
    tl.store($(candidate_vmul_35_ptr) + r343_z, r343_value)
    r344_row = tl.arange($(192))
    r344_col = tl.arange($(32))
    r344_lane = tl.arange($(32))
    for r344_group in range($(8)) {
      r344_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r344_group % $(8)) * $(6144)) + (r344_lane[None, :] * $(192)) + r344_row[:, None]) / $(6144)) % $(8)) * $(6144)) + ((((((r344_group % $(8)) * $(6144)) + (r344_lane[None, :] * $(192)) + r344_row[:, None]) / $(192)) % $(32)) * $(192)) + ((((r344_group % $(8)) * $(6144)) + (r344_lane[None, :] * $(192)) + r344_row[:, None]) % $(192))), dtype=tl.float32)
      r344_right = tl.load($(candidate_vmul_35_ptr) + (((((((r344_group % $(8)) * $(1024)) + (r344_lane[:, None] * $(32)) + r344_col[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r344_group % $(8)) * $(1024)) + (r344_lane[:, None] * $(32)) + r344_col[None, :]) / $(32)) % $(32)) * $(32)) + ((((r344_group % $(8)) * $(1024)) + (r344_lane[:, None] * $(32)) + r344_col[None, :]) % $(32))), dtype=tl.float32)
      r344_value = tl.dot(r344_left, r344_right)
      r344_value = r344_value.to(tl.float32)
      tl.store($(candidate_vbmm_68_ptr) + ((r344_group * $(6144)) + (r344_row[:, None] * $(32)) + r344_col[None, :]), r344_value)
    }
    r345_row = tl.arange($(32))
    r345_col = tl.arange($(192))
    r345_lane = tl.arange($(32))
    for r345_group in range($(8)) {
      r345_left = tl.load($(candidate_vmul_35_ptr) + (((((((r345_group % $(8)) * $(1024)) + (r345_row[:, None] * $(32)) + r345_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r345_group % $(8)) * $(1024)) + (r345_row[:, None] * $(32)) + r345_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r345_group % $(8)) * $(1024)) + (r345_row[:, None] * $(32)) + r345_lane[None, :]) % $(32))), dtype=tl.float32)
      r345_right = tl.load($(candidate_vto_copy_32_ptr) + (((((((r345_group % $(8)) * $(6144)) + (r345_col[None, :] * $(32)) + r345_lane[:, None]) / $(6144)) % $(8)) * $(6144)) + (((((r345_group % $(8)) * $(6144)) + (r345_col[None, :] * $(32)) + r345_lane[:, None]) % $(32)) * $(192)) + (((((r345_group % $(8)) * $(6144)) + (r345_col[None, :] * $(32)) + r345_lane[:, None]) / $(32)) % $(192))), dtype=tl.float32)
      r345_value = tl.dot(r345_left, r345_right)
      r345_value = r345_value.to(tl.float32)
      tl.store($(candidate_vbmm_69_ptr) + ((r345_group * $(6144)) + (r345_row[:, None] * $(192)) + r345_col[None, :]), r345_value)
    }
    r346_z = tl.arange($(98304))
    r346_value = tl.where(((((r346_z / $(1024)) % $(96)) >= $(0)) & (((r346_z / $(1024)) % $(96)) < $(32)) & ((((r346_z / $(1024)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_66_ptr) + ((((((((r346_z / $(128)) % $(8)) * $(4096)) + ((((r346_z / $(1024)) % $(96)) - $(0)) * $(128)) + (r346_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r346_z / $(128)) % $(8)) * $(32)) + (((r346_z / $(1024)) % $(96)) - $(0))) % $(32)) * $(128)) + (r346_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r346_value = r346_value.to(tl.bfloat16)
    r346_value = tl.broadcast_to(r346_value, [$(98304)])
    tl.store($(candidate_vslice_backward_35_ptr) + r346_z, r346_value)
    r347_z = tl.arange($(49152))
    r347_value = tl.load($(candidate_vbmm_68_ptr) + ((((r347_z / $(6144)) % $(8)) * $(6144)) + ((r347_z % $(192)) * $(32)) + ((r347_z / $(192)) % $(32))), dtype=tl.float32)
    r347_value = r347_value.to(tl.bfloat16)
    r347_value = tl.broadcast_to(r347_value, [$(49152)])
    tl.store($(candidate_vto_copy_65_ptr) + r347_z, r347_value)
    r348_z = tl.arange($(147456))
    r348_value = tl.where(((((r348_z / $(1536)) % $(96)) >= $(0)) & (((r348_z / $(1536)) % $(96)) < $(32)) & ((((r348_z / $(1536)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_65_ptr) + ((((r348_z / $(192)) % $(8)) * $(6144)) + (((((r348_z / $(1536)) % $(96)) - $(0)) % $(32)) * $(192)) + (r348_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r348_value = r348_value.to(tl.bfloat16)
    r348_value = tl.broadcast_to(r348_value, [$(147456)])
    tl.store($(candidate_vslice_backward_36_ptr) + r348_z, r348_value)
    r349_z = tl.arange($(49152))
    r349_value = tl.load($(candidate_vbmm_69_ptr) + ((((r349_z / $(6144)) % $(8)) * $(6144)) + (((r349_z / $(192)) % $(32)) * $(192)) + (r349_z % $(192))), dtype=tl.float32)
    r349_value = r349_value.to(tl.bfloat16)
    r349_value = tl.broadcast_to(r349_value, [$(49152)])
    tl.store($(candidate_vto_copy_66_ptr) + r349_z, r349_value)
    r350_z = tl.arange($(49152))
    r350_value = tl.where(((((r350_z / $(1536)) % $(32)) >= $(0)) & (((r350_z / $(1536)) % $(32)) < $(32)) & ((((r350_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_66_ptr) + ((((r350_z / $(192)) % $(8)) * $(6144)) + (((((r350_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(192)) + (r350_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r350_value = r350_value.to(tl.bfloat16)
    r350_value = tl.broadcast_to(r350_value, [$(49152)])
    tl.store($(candidate_vslice_backward_37_ptr) + r350_z, r350_value)
    r351_z = tl.arange($(65536))
    r351_value = tl.cast(0.0, tl.bfloat16)
    r351_value = r351_value.to(tl.bfloat16)
    r351_value = tl.broadcast_to(r351_value, [$(65536)])
    tl.store($(candidate_vfill_10_ptr) + r351_z, r351_value)
    r352_z = tl.arange($(98304))
    r352_value = tl.where(((((r352_z / $(1024)) % $(96)) >= $(32)) & (((r352_z / $(1024)) % $(96)) < $(96)) & ((((r352_z / $(1024)) % $(96)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vfill_10_ptr) + ((((((r352_z / $(1024)) % $(96)) - $(32)) % $(64)) * $(1024)) + (((r352_z / $(128)) % $(8)) * $(128)) + (r352_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_35_ptr) + ((((r352_z / $(1024)) % $(96)) * $(1024)) + (((r352_z / $(128)) % $(8)) * $(128)) + (r352_z % $(128))), dtype=tl.bfloat16))
    r352_value = r352_value.to(tl.bfloat16)
    r352_value = tl.broadcast_to(r352_value, [$(98304)])
    tl.store($(candidate_vslice_scatter_10_ptr) + r352_z, r352_value)
    r353_z = tl.arange($(98304))
    r353_value = tl.cast(0.0, tl.bfloat16)
    r353_value = r353_value.to(tl.bfloat16)
    r353_value = tl.broadcast_to(r353_value, [$(98304)])
    tl.store($(candidate_vfill_11_ptr) + r353_z, r353_value)
    r354_z = tl.arange($(147456))
    r354_value = tl.where(((((r354_z / $(1536)) % $(96)) >= $(32)) & (((r354_z / $(1536)) % $(96)) < $(96)) & ((((r354_z / $(1536)) % $(96)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vfill_11_ptr) + ((((((r354_z / $(1536)) % $(96)) - $(32)) % $(64)) * $(1536)) + (((r354_z / $(192)) % $(8)) * $(192)) + (r354_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_36_ptr) + ((((r354_z / $(1536)) % $(96)) * $(1536)) + (((r354_z / $(192)) % $(8)) * $(192)) + (r354_z % $(192))), dtype=tl.bfloat16))
    r354_value = r354_value.to(tl.bfloat16)
    r354_value = tl.broadcast_to(r354_value, [$(147456)])
    tl.store($(candidate_vslice_scatter_11_ptr) + r354_z, r354_value)
    r355_z = tl.arange($(6144))
    r355_lane = tl.arange($(8))
    r355_value = tl.sum(tl.load($(candidate_vslice_scatter_11_ptr) + ((((r355_z[:, None] / $(64)) % $(96)) * $(1536)) + (r355_lane[None, :] * $(192)) + (r355_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r355_value = r355_value.to(tl.bfloat16)
    r355_value = tl.broadcast_to(r355_value, [$(6144)])
    tl.store($(candidate_vsum_6_ptr) + r355_z, r355_value)
    r356_row = tl.arange($(96))
    r356_col = tl.arange($(512))
    r356_lane = tl.arange($(1024))
    for r356_group in range($(1)) {
      r356_left = tl.load($(candidate_vslice_scatter_10_ptr) + ((((((r356_row[:, None] * $(1024)) + r356_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + (((((r356_row[:, None] * $(1024)) + r356_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r356_row[:, None] * $(1024)) + r356_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r356_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r356_col[None, :] * $(1024)) + r356_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r356_col[None, :] * $(1024)) + r356_lane[:, None]) % $(128)) * $(512)) + ((((r356_col[None, :] * $(1024)) + r356_lane[:, None]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r356_value = tl.dot(r356_left, r356_right)
      r356_value = r356_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_70_ptr) + ((r356_group * $(49152)) + (r356_row[:, None] * $(512)) + r356_col[None, :]), r356_value)
    }
    r357_row = tl.arange($(96))
    r357_col = tl.arange($(512))
    r357_lane = tl.arange($(1024))
    for r357_group in range($(1)) {
      r357_left = tl.load($(candidate_vslice_scatter_11_ptr) + ((((((r357_row[:, None] * $(1024)) + r357_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + (((((r357_row[:, None] * $(1024)) + r357_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r357_row[:, None] * $(1024)) + r357_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r357_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r357_col[None, :] * $(1024)) + r357_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r357_col[None, :] * $(1024)) + r357_lane[:, None]) % $(128)) * $(512)) + ((((r357_col[None, :] * $(1024)) + r357_lane[:, None]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r357_value = tl.dot(r357_left, r357_right)
      r357_value = r357_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_71_ptr) + ((r357_group * $(49152)) + (r357_row[:, None] * $(512)) + r357_col[None, :]), r357_value)
    }
    r358_z = tl.arange($(49152))
    r358_value = (tl.load($(candidate_vbmm_70_ptr) + ((((r358_z / $(512)) % $(96)) * $(512)) + (r358_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_71_ptr) + ((((r358_z / $(512)) % $(96)) * $(512)) + (r358_z % $(512))), dtype=tl.bfloat16))
    r358_value = r358_value.to(tl.bfloat16)
    r358_value = tl.broadcast_to(r358_value, [$(49152)])
    tl.store($(candidate_vadd_36_ptr) + r358_z, r358_value)
    r359_z = tl.arange($(49152))
    r359_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_36_ptr) + ((((r359_z / $(512)) % $(96)) * $(512)) + (r359_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r359_value = r359_value.to(tl.bfloat16)
    r359_value = tl.broadcast_to(r359_value, [$(49152)])
    tl.store($(candidate_vselect_backward_5_ptr) + r359_z, r359_value)
    r360_z = tl.arange($(147456))
    r360_value = tl.where(((((r360_z / $(512)) % $(288)) >= $(96)) & (((r360_z / $(512)) % $(288)) < $(192)) & ((((r360_z / $(512)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_5_ptr) + ((((((r360_z / $(512)) % $(288)) - $(96)) % $(96)) * $(512)) + (r360_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r360_value = r360_value.to(tl.bfloat16)
    r360_value = tl.broadcast_to(r360_value, [$(147456)])
    tl.store($(candidate_vslice_backward_38_ptr) + r360_z, r360_value)
    r361_z = tl.arange($(147456))
    r361_value = (tl.load($(candidate_vadd_32_ptr) + ((((r361_z / $(512)) % $(288)) * $(512)) + (r361_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_38_ptr) + ((((r361_z / $(512)) % $(288)) * $(512)) + (r361_z % $(512))), dtype=tl.bfloat16))
    r361_value = r361_value.to(tl.bfloat16)
    r361_value = tl.broadcast_to(r361_value, [$(147456)])
    tl.store($(candidate_vadd_37_ptr) + r361_z, r361_value)
    r362_z = tl.arange($(18432))
    r362_value = tl.where(((((r362_z / $(64)) % $(288)) >= $(96)) & (((r362_z / $(64)) % $(288)) < $(192)) & ((((r362_z / $(64)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vsum_6_ptr) + ((((((r362_z / $(64)) % $(288)) - $(96)) % $(96)) * $(64)) + (r362_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r362_value = r362_value.to(tl.bfloat16)
    r362_value = tl.broadcast_to(r362_value, [$(18432)])
    tl.store($(candidate_vslice_backward_39_ptr) + r362_z, r362_value)
    r363_z = tl.arange($(18432))
    r363_value = (tl.load($(candidate_vadd_33_ptr) + ((((r363_z / $(64)) % $(288)) * $(64)) + (r363_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_39_ptr) + ((((r363_z / $(64)) % $(288)) * $(64)) + (r363_z % $(64))), dtype=tl.bfloat16))
    r363_value = r363_value.to(tl.bfloat16)
    r363_value = tl.broadcast_to(r363_value, [$(18432)])
    tl.store($(candidate_vadd_38_ptr) + r363_z, r363_value)
    r364_z = tl.arange($(294912))
    r364_value = tl.where(((((r364_z / $(1024)) % $(288)) >= $(96)) & (((r364_z / $(1024)) % $(288)) < $(128)) & ((((r364_z / $(1024)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_37_ptr) + ((((((r364_z / $(1024)) % $(288)) - $(96)) % $(32)) * $(1536)) + (((r364_z / $(128)) % $(8)) * $(192)) + (r364_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r364_value = r364_value.to(tl.bfloat16)
    r364_value = tl.broadcast_to(r364_value, [$(294912)])
    tl.store($(candidate_vslice_backward_40_ptr) + r364_z, r364_value)
    r365_z = tl.arange($(294912))
    r365_value = (tl.load($(candidate_vadd_34_ptr) + ((((r365_z / $(1024)) % $(288)) * $(1024)) + (((r365_z / $(128)) % $(8)) * $(128)) + (r365_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_40_ptr) + ((((r365_z / $(1024)) % $(288)) * $(1024)) + (((r365_z / $(128)) % $(8)) * $(128)) + (r365_z % $(128))), dtype=tl.bfloat16))
    r365_value = r365_value.to(tl.bfloat16)
    r365_value = tl.broadcast_to(r365_value, [$(294912)])
    tl.store($(candidate_vadd_39_ptr) + r365_z, r365_value)
    r366_z = tl.arange($(147456))
    r366_value = tl.where(((((r366_z / $(512)) % $(288)) >= $(96)) & (((r366_z / $(512)) % $(288)) < $(128)) & ((((r366_z / $(512)) % $(288)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_37_ptr) + ((((((r366_z / $(512)) % $(288)) - $(96)) % $(32)) * $(1536)) + (((r366_z / $(64)) % $(8)) * $(192)) + (r366_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r366_value = r366_value.to(tl.bfloat16)
    r366_value = tl.broadcast_to(r366_value, [$(147456)])
    tl.store($(candidate_vslice_backward_41_ptr) + r366_z, r366_value)
    r367_z = tl.arange($(147456))
    r367_value = (tl.load($(candidate_vadd_35_ptr) + ((((r367_z / $(512)) % $(288)) * $(512)) + (((r367_z / $(64)) % $(8)) * $(64)) + (r367_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_41_ptr) + ((((r367_z / $(512)) % $(288)) * $(512)) + (((r367_z / $(64)) % $(8)) * $(64)) + (r367_z % $(64))), dtype=tl.bfloat16))
    r367_value = r367_value.to(tl.bfloat16)
    r367_value = tl.broadcast_to(r367_value, [$(147456)])
    tl.store($(candidate_vadd_40_ptr) + r367_z, r367_value)
    r368_row = tl.arange($(96))
    r368_col = tl.arange($(128))
    r368_lane = tl.arange($(32))
    for r368_group in range($(8)) {
      r368_left = tl.load($(candidate_vto_copy_30_ptr) + (((((((r368_group % $(8)) * $(3072)) + (r368_lane[None, :] * $(96)) + r368_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r368_group % $(8)) * $(3072)) + (r368_lane[None, :] * $(96)) + r368_row[:, None]) / $(96)) % $(32)) * $(96)) + ((((r368_group % $(8)) * $(3072)) + (r368_lane[None, :] * $(96)) + r368_row[:, None]) % $(96))), dtype=tl.bfloat16)
      r368_right = tl.load($(candidate_vmm_4_ptr) + (((((((r368_group % $(8)) * $(4096)) + (r368_lane[:, None] * $(128)) + r368_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r368_group % $(8)) * $(4096)) + (r368_lane[:, None] * $(128)) + r368_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r368_group % $(8)) * $(4096)) + (r368_lane[:, None] * $(128)) + r368_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r368_value = tl.dot(r368_left, r368_right)
      r368_value = r368_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_72_ptr) + ((r368_group * $(12288)) + (r368_row[:, None] * $(128)) + r368_col[None, :]), r368_value)
    }
    r369_row = tl.arange($(32))
    r369_col = tl.arange($(96))
    r369_lane = tl.arange($(128))
    for r369_group in range($(8)) {
      r369_left = tl.load($(candidate_vmm_4_ptr) + (((((((r369_group % $(8)) * $(4096)) + (r369_row[:, None] * $(128)) + r369_lane[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r369_group % $(8)) * $(4096)) + (r369_row[:, None] * $(128)) + r369_lane[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r369_group % $(8)) * $(4096)) + (r369_row[:, None] * $(128)) + r369_lane[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r369_right = tl.load($(candidate_vbmm_9_ptr) + (((((((r369_group % $(8)) * $(12288)) + (r369_col[None, :] * $(128)) + r369_lane[:, None]) / $(128)) % $(96)) * $(1024)) + ((((((r369_group % $(8)) * $(12288)) + (r369_col[None, :] * $(128)) + r369_lane[:, None]) / $(12288)) % $(8)) * $(128)) + ((((r369_group % $(8)) * $(12288)) + (r369_col[None, :] * $(128)) + r369_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r369_value = tl.dot(r369_left, r369_right)
      r369_value = r369_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_73_ptr) + ((r369_group * $(3072)) + (r369_row[:, None] * $(96)) + r369_col[None, :]), r369_value)
    }
    r370_z = tl.arange($(24576))
    r370_value = tl.load($(candidate_vbmm_73_ptr) + ((((r370_z / $(3072)) % $(8)) * $(3072)) + (((r370_z / $(96)) % $(32)) * $(96)) + (r370_z % $(96))), dtype=tl.bfloat16)
    r370_value = r370_value.to(tl.float32)
    r370_value = tl.broadcast_to(r370_value, [$(24576)])
    tl.store($(candidate_vto_copy_67_ptr) + r370_z, r370_value)
    r371_z = tl.arange($(24576))
    r371_lane = tl.arange($(96))
    r371_row_values = tl.load($(candidate_vsoftmax_2_ptr) + ((((r371_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r371_z[:, None] / $(96)) % $(32)) * $(96)) + (r371_lane[None, :] % $(96))), dtype=tl.float32)
    r371_row_grad = tl.load($(candidate_vto_copy_67_ptr) + ((((r371_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r371_z[:, None] / $(96)) % $(32)) * $(96)) + (r371_lane[None, :] % $(96))), dtype=tl.float32)
    r371_weighted = r371_row_grad * r371_row_values
    r371_dot_grad = tl.sum(r371_weighted, axis=1)
    r371_value = (tl.load($(candidate_vto_copy_67_ptr) + ((((r371_z / $(3072)) % $(8)) * $(3072)) + (((r371_z / $(96)) % $(32)) * $(96)) + (r371_z % $(96))), dtype=tl.float32) - r371_dot_grad) * tl.load($(candidate_vsoftmax_2_ptr) + ((((r371_z / $(3072)) % $(8)) * $(3072)) + (((r371_z / $(96)) % $(32)) * $(96)) + (r371_z % $(96))), dtype=tl.float32)
    r371_value = r371_value.to(tl.float32)
    r371_value = tl.broadcast_to(r371_value, [$(24576)])
    tl.store($(candidate_vsoftmax_backward_data_6_ptr) + r371_z, r371_value)
    r372_z = tl.arange($(24576))
    r372_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r372_z / $(96)) % $(32)) * $(96)) + (r372_z % $(96))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_6_ptr) + ((((r372_z / $(3072)) % $(8)) * $(3072)) + (((r372_z / $(96)) % $(32)) * $(96)) + (r372_z % $(96))), dtype=tl.float32))
    r372_value = r372_value.to(tl.float32)
    r372_value = tl.broadcast_to(r372_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_15_ptr) + r372_z, r372_value)
    r373_z = tl.arange($(24576))
    r373_value = (tl.load($(candidate_vmasked_fill_15_ptr) + ((((r373_z / $(3072)) % $(8)) * $(3072)) + (((r373_z / $(96)) % $(32)) * $(96)) + (r373_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r373_value = r373_value.to(tl.float32)
    r373_value = tl.broadcast_to(r373_value, [$(24576)])
    tl.store($(candidate_vmul_36_ptr) + r373_z, r373_value)
    r374_row = tl.arange($(192))
    r374_col = tl.arange($(96))
    r374_lane = tl.arange($(32))
    for r374_group in range($(8)) {
      r374_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r374_group % $(8)) * $(6144)) + (r374_lane[None, :] * $(192)) + r374_row[:, None]) / $(6144)) % $(8)) * $(6144)) + ((((((r374_group % $(8)) * $(6144)) + (r374_lane[None, :] * $(192)) + r374_row[:, None]) / $(192)) % $(32)) * $(192)) + ((((r374_group % $(8)) * $(6144)) + (r374_lane[None, :] * $(192)) + r374_row[:, None]) % $(192))), dtype=tl.float32)
      r374_right = tl.load($(candidate_vmul_36_ptr) + (((((((r374_group % $(8)) * $(3072)) + (r374_lane[:, None] * $(96)) + r374_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r374_group % $(8)) * $(3072)) + (r374_lane[:, None] * $(96)) + r374_col[None, :]) / $(96)) % $(32)) * $(96)) + ((((r374_group % $(8)) * $(3072)) + (r374_lane[:, None] * $(96)) + r374_col[None, :]) % $(96))), dtype=tl.float32)
      r374_value = tl.dot(r374_left, r374_right)
      r374_value = r374_value.to(tl.float32)
      tl.store($(candidate_vbmm_74_ptr) + ((r374_group * $(18432)) + (r374_row[:, None] * $(96)) + r374_col[None, :]), r374_value)
    }
    r375_row = tl.arange($(32))
    r375_col = tl.arange($(192))
    r375_lane = tl.arange($(96))
    for r375_group in range($(8)) {
      r375_left = tl.load($(candidate_vmul_36_ptr) + (((((((r375_group % $(8)) * $(3072)) + (r375_row[:, None] * $(96)) + r375_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r375_group % $(8)) * $(3072)) + (r375_row[:, None] * $(96)) + r375_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r375_group % $(8)) * $(3072)) + (r375_row[:, None] * $(96)) + r375_lane[None, :]) % $(96))), dtype=tl.float32)
      r375_right = tl.load($(candidate_vto_copy_29_ptr) + (((((((r375_group % $(8)) * $(18432)) + (r375_col[None, :] * $(96)) + r375_lane[:, None]) / $(18432)) % $(8)) * $(18432)) + (((((r375_group % $(8)) * $(18432)) + (r375_col[None, :] * $(96)) + r375_lane[:, None]) % $(96)) * $(192)) + (((((r375_group % $(8)) * $(18432)) + (r375_col[None, :] * $(96)) + r375_lane[:, None]) / $(96)) % $(192))), dtype=tl.float32)
      r375_value = tl.dot(r375_left, r375_right)
      r375_value = r375_value.to(tl.float32)
      tl.store($(candidate_vbmm_75_ptr) + ((r375_group * $(6144)) + (r375_row[:, None] * $(192)) + r375_col[None, :]), r375_value)
    }
    r376_z = tl.arange($(98304))
    r376_value = tl.where(((((r376_z / $(1024)) % $(96)) >= $(0)) & (((r376_z / $(1024)) % $(96)) < $(96)) & ((((r376_z / $(1024)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_72_ptr) + ((((((((r376_z / $(128)) % $(8)) * $(12288)) + ((((r376_z / $(1024)) % $(96)) - $(0)) * $(128)) + (r376_z % $(128))) / $(12288)) % $(8)) * $(12288)) + ((((((r376_z / $(128)) % $(8)) * $(96)) + (((r376_z / $(1024)) % $(96)) - $(0))) % $(96)) * $(128)) + (r376_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r376_value = r376_value.to(tl.bfloat16)
    r376_value = tl.broadcast_to(r376_value, [$(98304)])
    tl.store($(candidate_vslice_backward_42_ptr) + r376_z, r376_value)
    r377_z = tl.arange($(147456))
    r377_value = tl.load($(candidate_vbmm_74_ptr) + ((((r377_z / $(18432)) % $(8)) * $(18432)) + ((r377_z % $(192)) * $(96)) + ((r377_z / $(192)) % $(96))), dtype=tl.float32)
    r377_value = r377_value.to(tl.bfloat16)
    r377_value = tl.broadcast_to(r377_value, [$(147456)])
    tl.store($(candidate_vto_copy_68_ptr) + r377_z, r377_value)
    r378_z = tl.arange($(147456))
    r378_value = tl.where(((((r378_z / $(1536)) % $(96)) >= $(0)) & (((r378_z / $(1536)) % $(96)) < $(96)) & ((((r378_z / $(1536)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_68_ptr) + ((((r378_z / $(192)) % $(8)) * $(18432)) + (((((r378_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(192)) + (r378_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r378_value = r378_value.to(tl.bfloat16)
    r378_value = tl.broadcast_to(r378_value, [$(147456)])
    tl.store($(candidate_vslice_backward_43_ptr) + r378_z, r378_value)
    r379_z = tl.arange($(49152))
    r379_value = tl.load($(candidate_vbmm_75_ptr) + ((((r379_z / $(6144)) % $(8)) * $(6144)) + (((r379_z / $(192)) % $(32)) * $(192)) + (r379_z % $(192))), dtype=tl.float32)
    r379_value = r379_value.to(tl.bfloat16)
    r379_value = tl.broadcast_to(r379_value, [$(49152)])
    tl.store($(candidate_vto_copy_69_ptr) + r379_z, r379_value)
    r380_z = tl.arange($(49152))
    r380_value = tl.where(((((r380_z / $(1536)) % $(32)) >= $(0)) & (((r380_z / $(1536)) % $(32)) < $(32)) & ((((r380_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_69_ptr) + ((((r380_z / $(192)) % $(8)) * $(6144)) + (((((r380_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(192)) + (r380_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r380_value = r380_value.to(tl.bfloat16)
    r380_value = tl.broadcast_to(r380_value, [$(49152)])
    tl.store($(candidate_vslice_backward_44_ptr) + r380_z, r380_value)
    r381_z = tl.arange($(0))
    r381_value = tl.cast(0.0, tl.bfloat16)
    r381_value = r381_value.to(tl.bfloat16)
    r381_value = tl.broadcast_to(r381_value, [$(0)])
    tl.store($(candidate_vfill_12_ptr) + r381_z, r381_value)
    r382_z = tl.arange($(98304))
    r382_value = tl.where(((((r382_z / $(1024)) % $(96)) >= $(96)) & (((r382_z / $(1024)) % $(96)) < $(96)) & ((((r382_z / $(1024)) % $(96)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_12_ptr) + ((((((r382_z / $(1024)) % $(96)) - $(96)) % $(0)) * $(1024)) + (((r382_z / $(128)) % $(8)) * $(128)) + (r382_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_42_ptr) + ((((r382_z / $(1024)) % $(96)) * $(1024)) + (((r382_z / $(128)) % $(8)) * $(128)) + (r382_z % $(128))), dtype=tl.bfloat16))
    r382_value = r382_value.to(tl.bfloat16)
    r382_value = tl.broadcast_to(r382_value, [$(98304)])
    tl.store($(candidate_vslice_scatter_12_ptr) + r382_z, r382_value)
    r383_z = tl.arange($(0))
    r383_value = tl.cast(0.0, tl.bfloat16)
    r383_value = r383_value.to(tl.bfloat16)
    r383_value = tl.broadcast_to(r383_value, [$(0)])
    tl.store($(candidate_vfill_13_ptr) + r383_z, r383_value)
    r384_z = tl.arange($(147456))
    r384_value = tl.where(((((r384_z / $(1536)) % $(96)) >= $(96)) & (((r384_z / $(1536)) % $(96)) < $(96)) & ((((r384_z / $(1536)) % $(96)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_13_ptr) + ((((((r384_z / $(1536)) % $(96)) - $(96)) % $(0)) * $(1536)) + (((r384_z / $(192)) % $(8)) * $(192)) + (r384_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_43_ptr) + ((((r384_z / $(1536)) % $(96)) * $(1536)) + (((r384_z / $(192)) % $(8)) * $(192)) + (r384_z % $(192))), dtype=tl.bfloat16))
    r384_value = r384_value.to(tl.bfloat16)
    r384_value = tl.broadcast_to(r384_value, [$(147456)])
    tl.store($(candidate_vslice_scatter_13_ptr) + r384_z, r384_value)
    r385_z = tl.arange($(6144))
    r385_lane = tl.arange($(8))
    r385_value = tl.sum(tl.load($(candidate_vslice_scatter_13_ptr) + ((((r385_z[:, None] / $(64)) % $(96)) * $(1536)) + (r385_lane[None, :] * $(192)) + (r385_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r385_value = r385_value.to(tl.bfloat16)
    r385_value = tl.broadcast_to(r385_value, [$(6144)])
    tl.store($(candidate_vsum_7_ptr) + r385_z, r385_value)
    r386_row = tl.arange($(96))
    r386_col = tl.arange($(512))
    r386_lane = tl.arange($(1024))
    for r386_group in range($(1)) {
      r386_left = tl.load($(candidate_vslice_scatter_12_ptr) + ((((((r386_row[:, None] * $(1024)) + r386_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + (((((r386_row[:, None] * $(1024)) + r386_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r386_row[:, None] * $(1024)) + r386_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r386_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r386_col[None, :] * $(1024)) + r386_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r386_col[None, :] * $(1024)) + r386_lane[:, None]) % $(128)) * $(512)) + ((((r386_col[None, :] * $(1024)) + r386_lane[:, None]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r386_value = tl.dot(r386_left, r386_right)
      r386_value = r386_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_76_ptr) + ((r386_group * $(49152)) + (r386_row[:, None] * $(512)) + r386_col[None, :]), r386_value)
    }
    r387_row = tl.arange($(96))
    r387_col = tl.arange($(512))
    r387_lane = tl.arange($(1024))
    for r387_group in range($(1)) {
      r387_left = tl.load($(candidate_vslice_scatter_13_ptr) + ((((((r387_row[:, None] * $(1024)) + r387_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + (((((r387_row[:, None] * $(1024)) + r387_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r387_row[:, None] * $(1024)) + r387_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r387_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r387_col[None, :] * $(1024)) + r387_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r387_col[None, :] * $(1024)) + r387_lane[:, None]) % $(128)) * $(512)) + ((((r387_col[None, :] * $(1024)) + r387_lane[:, None]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r387_value = tl.dot(r387_left, r387_right)
      r387_value = r387_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_77_ptr) + ((r387_group * $(49152)) + (r387_row[:, None] * $(512)) + r387_col[None, :]), r387_value)
    }
    r388_z = tl.arange($(49152))
    r388_value = (tl.load($(candidate_vbmm_76_ptr) + ((((r388_z / $(512)) % $(96)) * $(512)) + (r388_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_77_ptr) + ((((r388_z / $(512)) % $(96)) * $(512)) + (r388_z % $(512))), dtype=tl.bfloat16))
    r388_value = r388_value.to(tl.bfloat16)
    r388_value = tl.broadcast_to(r388_value, [$(49152)])
    tl.store($(candidate_vadd_41_ptr) + r388_z, r388_value)
    r389_z = tl.arange($(49152))
    r389_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_41_ptr) + ((((r389_z / $(512)) % $(96)) * $(512)) + (r389_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r389_value = r389_value.to(tl.bfloat16)
    r389_value = tl.broadcast_to(r389_value, [$(49152)])
    tl.store($(candidate_vselect_backward_6_ptr) + r389_z, r389_value)
    r390_z = tl.arange($(147456))
    r390_value = tl.where(((((r390_z / $(512)) % $(288)) >= $(0)) & (((r390_z / $(512)) % $(288)) < $(96)) & ((((r390_z / $(512)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_6_ptr) + ((((((r390_z / $(512)) % $(288)) - $(0)) % $(96)) * $(512)) + (r390_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r390_value = r390_value.to(tl.bfloat16)
    r390_value = tl.broadcast_to(r390_value, [$(147456)])
    tl.store($(candidate_vslice_backward_45_ptr) + r390_z, r390_value)
    r391_z = tl.arange($(147456))
    r391_value = (tl.load($(candidate_vadd_37_ptr) + ((((r391_z / $(512)) % $(288)) * $(512)) + (r391_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_45_ptr) + ((((r391_z / $(512)) % $(288)) * $(512)) + (r391_z % $(512))), dtype=tl.bfloat16))
    r391_value = r391_value.to(tl.bfloat16)
    r391_value = tl.broadcast_to(r391_value, [$(147456)])
    tl.store($(candidate_vadd_42_ptr) + r391_z, r391_value)
    r392_z = tl.arange($(18432))
    r392_value = tl.where(((((r392_z / $(64)) % $(288)) >= $(0)) & (((r392_z / $(64)) % $(288)) < $(96)) & ((((r392_z / $(64)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_7_ptr) + ((((((r392_z / $(64)) % $(288)) - $(0)) % $(96)) * $(64)) + (r392_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r392_value = r392_value.to(tl.bfloat16)
    r392_value = tl.broadcast_to(r392_value, [$(18432)])
    tl.store($(candidate_vslice_backward_46_ptr) + r392_z, r392_value)
    r393_z = tl.arange($(18432))
    r393_value = (tl.load($(candidate_vadd_38_ptr) + ((((r393_z / $(64)) % $(288)) * $(64)) + (r393_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_46_ptr) + ((((r393_z / $(64)) % $(288)) * $(64)) + (r393_z % $(64))), dtype=tl.bfloat16))
    r393_value = r393_value.to(tl.bfloat16)
    r393_value = tl.broadcast_to(r393_value, [$(18432)])
    tl.store($(candidate_vadd_43_ptr) + r393_z, r393_value)
    r394_z = tl.arange($(294912))
    r394_value = tl.where(((((r394_z / $(1024)) % $(288)) >= $(64)) & (((r394_z / $(1024)) % $(288)) < $(96)) & ((((r394_z / $(1024)) % $(288)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_44_ptr) + ((((((r394_z / $(1024)) % $(288)) - $(64)) % $(32)) * $(1536)) + (((r394_z / $(128)) % $(8)) * $(192)) + (r394_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r394_value = r394_value.to(tl.bfloat16)
    r394_value = tl.broadcast_to(r394_value, [$(294912)])
    tl.store($(candidate_vslice_backward_47_ptr) + r394_z, r394_value)
    r395_z = tl.arange($(294912))
    r395_value = (tl.load($(candidate_vadd_39_ptr) + ((((r395_z / $(1024)) % $(288)) * $(1024)) + (((r395_z / $(128)) % $(8)) * $(128)) + (r395_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_47_ptr) + ((((r395_z / $(1024)) % $(288)) * $(1024)) + (((r395_z / $(128)) % $(8)) * $(128)) + (r395_z % $(128))), dtype=tl.bfloat16))
    r395_value = r395_value.to(tl.bfloat16)
    r395_value = tl.broadcast_to(r395_value, [$(294912)])
    tl.store($(candidate_vadd_44_ptr) + r395_z, r395_value)
    r396_z = tl.arange($(147456))
    r396_value = tl.where(((((r396_z / $(512)) % $(288)) >= $(64)) & (((r396_z / $(512)) % $(288)) < $(96)) & ((((r396_z / $(512)) % $(288)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_44_ptr) + ((((((r396_z / $(512)) % $(288)) - $(64)) % $(32)) * $(1536)) + (((r396_z / $(64)) % $(8)) * $(192)) + (r396_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r396_value = r396_value.to(tl.bfloat16)
    r396_value = tl.broadcast_to(r396_value, [$(147456)])
    tl.store($(candidate_vslice_backward_48_ptr) + r396_z, r396_value)
    r397_z = tl.arange($(147456))
    r397_value = (tl.load($(candidate_vadd_40_ptr) + ((((r397_z / $(512)) % $(288)) * $(512)) + (((r397_z / $(64)) % $(8)) * $(64)) + (r397_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_48_ptr) + ((((r397_z / $(512)) % $(288)) * $(512)) + (((r397_z / $(64)) % $(8)) * $(64)) + (r397_z % $(64))), dtype=tl.bfloat16))
    r397_value = r397_value.to(tl.bfloat16)
    r397_value = tl.broadcast_to(r397_value, [$(147456)])
    tl.store($(candidate_vadd_45_ptr) + r397_z, r397_value)
    r398_row = tl.arange($(64))
    r398_col = tl.arange($(128))
    r398_lane = tl.arange($(32))
    for r398_group in range($(8)) {
      r398_left = tl.load($(candidate_vto_copy_27_ptr) + (((((((r398_group % $(8)) * $(2048)) + (r398_lane[None, :] * $(64)) + r398_row[:, None]) / $(2048)) % $(8)) * $(2048)) + ((((((r398_group % $(8)) * $(2048)) + (r398_lane[None, :] * $(64)) + r398_row[:, None]) / $(64)) % $(32)) * $(64)) + ((((r398_group % $(8)) * $(2048)) + (r398_lane[None, :] * $(64)) + r398_row[:, None]) % $(64))), dtype=tl.bfloat16)
      r398_right = tl.load($(candidate_vmm_4_ptr) + (((((((r398_group % $(8)) * $(4096)) + (r398_lane[:, None] * $(128)) + r398_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r398_group % $(8)) * $(4096)) + (r398_lane[:, None] * $(128)) + r398_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r398_group % $(8)) * $(4096)) + (r398_lane[:, None] * $(128)) + r398_col[None, :]) % $(128)) + $(32768)), dtype=tl.bfloat16)
      r398_value = tl.dot(r398_left, r398_right)
      r398_value = r398_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_78_ptr) + ((r398_group * $(8192)) + (r398_row[:, None] * $(128)) + r398_col[None, :]), r398_value)
    }
    r399_row = tl.arange($(32))
    r399_col = tl.arange($(64))
    r399_lane = tl.arange($(128))
    for r399_group in range($(8)) {
      r399_left = tl.load($(candidate_vmm_4_ptr) + (((((((r399_group % $(8)) * $(4096)) + (r399_row[:, None] * $(128)) + r399_lane[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r399_group % $(8)) * $(4096)) + (r399_row[:, None] * $(128)) + r399_lane[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r399_group % $(8)) * $(4096)) + (r399_row[:, None] * $(128)) + r399_lane[None, :]) % $(128)) + $(32768)), dtype=tl.bfloat16)
      r399_right = tl.load($(candidate_vbmm_5_ptr) + (((((((r399_group % $(8)) * $(8192)) + (r399_col[None, :] * $(128)) + r399_lane[:, None]) / $(128)) % $(64)) * $(1024)) + ((((((r399_group % $(8)) * $(8192)) + (r399_col[None, :] * $(128)) + r399_lane[:, None]) / $(8192)) % $(8)) * $(128)) + ((((r399_group % $(8)) * $(8192)) + (r399_col[None, :] * $(128)) + r399_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r399_value = tl.dot(r399_left, r399_right)
      r399_value = r399_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_79_ptr) + ((r399_group * $(2048)) + (r399_row[:, None] * $(64)) + r399_col[None, :]), r399_value)
    }
    r400_z = tl.arange($(16384))
    r400_value = tl.load($(candidate_vbmm_79_ptr) + ((((r400_z / $(2048)) % $(8)) * $(2048)) + (((r400_z / $(64)) % $(32)) * $(64)) + (r400_z % $(64))), dtype=tl.bfloat16)
    r400_value = r400_value.to(tl.float32)
    r400_value = tl.broadcast_to(r400_value, [$(16384)])
    tl.store($(candidate_vto_copy_70_ptr) + r400_z, r400_value)
    r401_z = tl.arange($(16384))
    r401_lane = tl.arange($(64))
    r401_row_values = tl.load($(candidate_vsoftmax_1_ptr) + ((((r401_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r401_z[:, None] / $(64)) % $(32)) * $(64)) + (r401_lane[None, :] % $(64))), dtype=tl.float32)
    r401_row_grad = tl.load($(candidate_vto_copy_70_ptr) + ((((r401_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r401_z[:, None] / $(64)) % $(32)) * $(64)) + (r401_lane[None, :] % $(64))), dtype=tl.float32)
    r401_weighted = r401_row_grad * r401_row_values
    r401_dot_grad = tl.sum(r401_weighted, axis=1)
    r401_value = (tl.load($(candidate_vto_copy_70_ptr) + ((((r401_z / $(2048)) % $(8)) * $(2048)) + (((r401_z / $(64)) % $(32)) * $(64)) + (r401_z % $(64))), dtype=tl.float32) - r401_dot_grad) * tl.load($(candidate_vsoftmax_1_ptr) + ((((r401_z / $(2048)) % $(8)) * $(2048)) + (((r401_z / $(64)) % $(32)) * $(64)) + (r401_z % $(64))), dtype=tl.float32)
    r401_value = r401_value.to(tl.float32)
    r401_value = tl.broadcast_to(r401_value, [$(16384)])
    tl.store($(candidate_vsoftmax_backward_data_7_ptr) + r401_z, r401_value)
    r402_z = tl.arange($(16384))
    r402_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r402_z / $(64)) % $(32)) * $(64)) + (r402_z % $(64))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_7_ptr) + ((((r402_z / $(2048)) % $(8)) * $(2048)) + (((r402_z / $(64)) % $(32)) * $(64)) + (r402_z % $(64))), dtype=tl.float32))
    r402_value = r402_value.to(tl.float32)
    r402_value = tl.broadcast_to(r402_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_16_ptr) + r402_z, r402_value)
    r403_z = tl.arange($(16384))
    r403_value = (tl.load($(candidate_vmasked_fill_16_ptr) + ((((r403_z / $(2048)) % $(8)) * $(2048)) + (((r403_z / $(64)) % $(32)) * $(64)) + (r403_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r403_value = r403_value.to(tl.float32)
    r403_value = tl.broadcast_to(r403_value, [$(16384)])
    tl.store($(candidate_vmul_37_ptr) + r403_z, r403_value)
    r404_row = tl.arange($(192))
    r404_col = tl.arange($(64))
    r404_lane = tl.arange($(32))
    for r404_group in range($(8)) {
      r404_left = tl.load($(candidate_vto_copy_25_ptr) + (((((((r404_group % $(8)) * $(6144)) + (r404_lane[None, :] * $(192)) + r404_row[:, None]) / $(6144)) % $(8)) * $(6144)) + ((((((r404_group % $(8)) * $(6144)) + (r404_lane[None, :] * $(192)) + r404_row[:, None]) / $(192)) % $(32)) * $(192)) + ((((r404_group % $(8)) * $(6144)) + (r404_lane[None, :] * $(192)) + r404_row[:, None]) % $(192))), dtype=tl.float32)
      r404_right = tl.load($(candidate_vmul_37_ptr) + (((((((r404_group % $(8)) * $(2048)) + (r404_lane[:, None] * $(64)) + r404_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r404_group % $(8)) * $(2048)) + (r404_lane[:, None] * $(64)) + r404_col[None, :]) / $(64)) % $(32)) * $(64)) + ((((r404_group % $(8)) * $(2048)) + (r404_lane[:, None] * $(64)) + r404_col[None, :]) % $(64))), dtype=tl.float32)
      r404_value = tl.dot(r404_left, r404_right)
      r404_value = r404_value.to(tl.float32)
      tl.store($(candidate_vbmm_80_ptr) + ((r404_group * $(12288)) + (r404_row[:, None] * $(64)) + r404_col[None, :]), r404_value)
    }
    r405_row = tl.arange($(32))
    r405_col = tl.arange($(192))
    r405_lane = tl.arange($(64))
    for r405_group in range($(8)) {
      r405_left = tl.load($(candidate_vmul_37_ptr) + (((((((r405_group % $(8)) * $(2048)) + (r405_row[:, None] * $(64)) + r405_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r405_group % $(8)) * $(2048)) + (r405_row[:, None] * $(64)) + r405_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r405_group % $(8)) * $(2048)) + (r405_row[:, None] * $(64)) + r405_lane[None, :]) % $(64))), dtype=tl.float32)
      r405_right = tl.load($(candidate_vto_copy_26_ptr) + (((((((r405_group % $(8)) * $(12288)) + (r405_col[None, :] * $(64)) + r405_lane[:, None]) / $(12288)) % $(8)) * $(12288)) + (((((r405_group % $(8)) * $(12288)) + (r405_col[None, :] * $(64)) + r405_lane[:, None]) % $(64)) * $(192)) + (((((r405_group % $(8)) * $(12288)) + (r405_col[None, :] * $(64)) + r405_lane[:, None]) / $(64)) % $(192))), dtype=tl.float32)
      r405_value = tl.dot(r405_left, r405_right)
      r405_value = r405_value.to(tl.float32)
      tl.store($(candidate_vbmm_81_ptr) + ((r405_group * $(6144)) + (r405_row[:, None] * $(192)) + r405_col[None, :]), r405_value)
    }
    r406_z = tl.arange($(98304))
    r406_value = tl.where(((((r406_z / $(1024)) % $(96)) >= $(0)) & (((r406_z / $(1024)) % $(96)) < $(64)) & ((((r406_z / $(1024)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_78_ptr) + ((((((((r406_z / $(128)) % $(8)) * $(8192)) + ((((r406_z / $(1024)) % $(96)) - $(0)) * $(128)) + (r406_z % $(128))) / $(8192)) % $(8)) * $(8192)) + ((((((r406_z / $(128)) % $(8)) * $(64)) + (((r406_z / $(1024)) % $(96)) - $(0))) % $(64)) * $(128)) + (r406_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r406_value = r406_value.to(tl.bfloat16)
    r406_value = tl.broadcast_to(r406_value, [$(98304)])
    tl.store($(candidate_vslice_backward_49_ptr) + r406_z, r406_value)
    r407_z = tl.arange($(98304))
    r407_value = tl.load($(candidate_vbmm_80_ptr) + ((((r407_z / $(12288)) % $(8)) * $(12288)) + ((r407_z % $(192)) * $(64)) + ((r407_z / $(192)) % $(64))), dtype=tl.float32)
    r407_value = r407_value.to(tl.bfloat16)
    r407_value = tl.broadcast_to(r407_value, [$(98304)])
    tl.store($(candidate_vto_copy_71_ptr) + r407_z, r407_value)
    r408_z = tl.arange($(147456))
    r408_value = tl.where(((((r408_z / $(1536)) % $(96)) >= $(0)) & (((r408_z / $(1536)) % $(96)) < $(64)) & ((((r408_z / $(1536)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_71_ptr) + ((((r408_z / $(192)) % $(8)) * $(12288)) + (((((r408_z / $(1536)) % $(96)) - $(0)) % $(64)) * $(192)) + (r408_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r408_value = r408_value.to(tl.bfloat16)
    r408_value = tl.broadcast_to(r408_value, [$(147456)])
    tl.store($(candidate_vslice_backward_50_ptr) + r408_z, r408_value)
    r409_z = tl.arange($(49152))
    r409_value = tl.load($(candidate_vbmm_81_ptr) + ((((r409_z / $(6144)) % $(8)) * $(6144)) + (((r409_z / $(192)) % $(32)) * $(192)) + (r409_z % $(192))), dtype=tl.float32)
    r409_value = r409_value.to(tl.bfloat16)
    r409_value = tl.broadcast_to(r409_value, [$(49152)])
    tl.store($(candidate_vto_copy_72_ptr) + r409_z, r409_value)
    r410_z = tl.arange($(49152))
    r410_value = tl.where(((((r410_z / $(1536)) % $(32)) >= $(0)) & (((r410_z / $(1536)) % $(32)) < $(32)) & ((((r410_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_72_ptr) + ((((r410_z / $(192)) % $(8)) * $(6144)) + (((((r410_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(192)) + (r410_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r410_value = r410_value.to(tl.bfloat16)
    r410_value = tl.broadcast_to(r410_value, [$(49152)])
    tl.store($(candidate_vslice_backward_51_ptr) + r410_z, r410_value)
    r411_z = tl.arange($(32768))
    r411_value = tl.cast(0.0, tl.bfloat16)
    r411_value = r411_value.to(tl.bfloat16)
    r411_value = tl.broadcast_to(r411_value, [$(32768)])
    tl.store($(candidate_vfill_14_ptr) + r411_z, r411_value)
    r412_z = tl.arange($(98304))
    r412_value = tl.where(((((r412_z / $(1024)) % $(96)) >= $(64)) & (((r412_z / $(1024)) % $(96)) < $(96)) & ((((r412_z / $(1024)) % $(96)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_14_ptr) + ((((((r412_z / $(1024)) % $(96)) - $(64)) % $(32)) * $(1024)) + (((r412_z / $(128)) % $(8)) * $(128)) + (r412_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_49_ptr) + ((((r412_z / $(1024)) % $(96)) * $(1024)) + (((r412_z / $(128)) % $(8)) * $(128)) + (r412_z % $(128))), dtype=tl.bfloat16))
    r412_value = r412_value.to(tl.bfloat16)
    r412_value = tl.broadcast_to(r412_value, [$(98304)])
    tl.store($(candidate_vslice_scatter_14_ptr) + r412_z, r412_value)
    r413_z = tl.arange($(49152))
    r413_value = tl.cast(0.0, tl.bfloat16)
    r413_value = r413_value.to(tl.bfloat16)
    r413_value = tl.broadcast_to(r413_value, [$(49152)])
    tl.store($(candidate_vfill_15_ptr) + r413_z, r413_value)
    r414_z = tl.arange($(147456))
    r414_value = tl.where(((((r414_z / $(1536)) % $(96)) >= $(64)) & (((r414_z / $(1536)) % $(96)) < $(96)) & ((((r414_z / $(1536)) % $(96)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_15_ptr) + ((((((r414_z / $(1536)) % $(96)) - $(64)) % $(32)) * $(1536)) + (((r414_z / $(192)) % $(8)) * $(192)) + (r414_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_50_ptr) + ((((r414_z / $(1536)) % $(96)) * $(1536)) + (((r414_z / $(192)) % $(8)) * $(192)) + (r414_z % $(192))), dtype=tl.bfloat16))
    r414_value = r414_value.to(tl.bfloat16)
    r414_value = tl.broadcast_to(r414_value, [$(147456)])
    tl.store($(candidate_vslice_scatter_15_ptr) + r414_z, r414_value)
    r415_z = tl.arange($(6144))
    r415_lane = tl.arange($(8))
    r415_value = tl.sum(tl.load($(candidate_vslice_scatter_15_ptr) + ((((r415_z[:, None] / $(64)) % $(96)) * $(1536)) + (r415_lane[None, :] * $(192)) + (r415_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r415_value = r415_value.to(tl.bfloat16)
    r415_value = tl.broadcast_to(r415_value, [$(6144)])
    tl.store($(candidate_vsum_8_ptr) + r415_z, r415_value)
    r416_row = tl.arange($(96))
    r416_col = tl.arange($(512))
    r416_lane = tl.arange($(1024))
    for r416_group in range($(1)) {
      r416_left = tl.load($(candidate_vslice_scatter_14_ptr) + ((((((r416_row[:, None] * $(1024)) + r416_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + (((((r416_row[:, None] * $(1024)) + r416_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r416_row[:, None] * $(1024)) + r416_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r416_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r416_col[None, :] * $(1024)) + r416_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r416_col[None, :] * $(1024)) + r416_lane[:, None]) % $(128)) * $(512)) + ((((r416_col[None, :] * $(1024)) + r416_lane[:, None]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r416_value = tl.dot(r416_left, r416_right)
      r416_value = r416_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_82_ptr) + ((r416_group * $(49152)) + (r416_row[:, None] * $(512)) + r416_col[None, :]), r416_value)
    }
    r417_row = tl.arange($(96))
    r417_col = tl.arange($(512))
    r417_lane = tl.arange($(1024))
    for r417_group in range($(1)) {
      r417_left = tl.load($(candidate_vslice_scatter_15_ptr) + ((((((r417_row[:, None] * $(1024)) + r417_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + (((((r417_row[:, None] * $(1024)) + r417_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r417_row[:, None] * $(1024)) + r417_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r417_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r417_col[None, :] * $(1024)) + r417_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r417_col[None, :] * $(1024)) + r417_lane[:, None]) % $(128)) * $(512)) + ((((r417_col[None, :] * $(1024)) + r417_lane[:, None]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r417_value = tl.dot(r417_left, r417_right)
      r417_value = r417_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_83_ptr) + ((r417_group * $(49152)) + (r417_row[:, None] * $(512)) + r417_col[None, :]), r417_value)
    }
    r418_z = tl.arange($(49152))
    r418_value = (tl.load($(candidate_vbmm_82_ptr) + ((((r418_z / $(512)) % $(96)) * $(512)) + (r418_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_83_ptr) + ((((r418_z / $(512)) % $(96)) * $(512)) + (r418_z % $(512))), dtype=tl.bfloat16))
    r418_value = r418_value.to(tl.bfloat16)
    r418_value = tl.broadcast_to(r418_value, [$(49152)])
    tl.store($(candidate_vadd_46_ptr) + r418_z, r418_value)
    r419_z = tl.arange($(49152))
    r419_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_46_ptr) + ((((r419_z / $(512)) % $(96)) * $(512)) + (r419_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r419_value = r419_value.to(tl.bfloat16)
    r419_value = tl.broadcast_to(r419_value, [$(49152)])
    tl.store($(candidate_vselect_backward_7_ptr) + r419_z, r419_value)
    r420_z = tl.arange($(147456))
    r420_value = tl.where(((((r420_z / $(512)) % $(288)) >= $(0)) & (((r420_z / $(512)) % $(288)) < $(96)) & ((((r420_z / $(512)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_7_ptr) + ((((((r420_z / $(512)) % $(288)) - $(0)) % $(96)) * $(512)) + (r420_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r420_value = r420_value.to(tl.bfloat16)
    r420_value = tl.broadcast_to(r420_value, [$(147456)])
    tl.store($(candidate_vslice_backward_52_ptr) + r420_z, r420_value)
    r421_z = tl.arange($(147456))
    r421_value = (tl.load($(candidate_vadd_42_ptr) + ((((r421_z / $(512)) % $(288)) * $(512)) + (r421_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_52_ptr) + ((((r421_z / $(512)) % $(288)) * $(512)) + (r421_z % $(512))), dtype=tl.bfloat16))
    r421_value = r421_value.to(tl.bfloat16)
    r421_value = tl.broadcast_to(r421_value, [$(147456)])
    tl.store($(candidate_vadd_47_ptr) + r421_z, r421_value)
    r422_z = tl.arange($(18432))
    r422_value = tl.where(((((r422_z / $(64)) % $(288)) >= $(0)) & (((r422_z / $(64)) % $(288)) < $(96)) & ((((r422_z / $(64)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_8_ptr) + ((((((r422_z / $(64)) % $(288)) - $(0)) % $(96)) * $(64)) + (r422_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r422_value = r422_value.to(tl.bfloat16)
    r422_value = tl.broadcast_to(r422_value, [$(18432)])
    tl.store($(candidate_vslice_backward_53_ptr) + r422_z, r422_value)
    r423_z = tl.arange($(18432))
    r423_value = (tl.load($(candidate_vadd_43_ptr) + ((((r423_z / $(64)) % $(288)) * $(64)) + (r423_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_53_ptr) + ((((r423_z / $(64)) % $(288)) * $(64)) + (r423_z % $(64))), dtype=tl.bfloat16))
    r423_value = r423_value.to(tl.bfloat16)
    r423_value = tl.broadcast_to(r423_value, [$(18432)])
    tl.store($(candidate_vadd_48_ptr) + r423_z, r423_value)
    r424_z = tl.arange($(294912))
    r424_value = tl.where(((((r424_z / $(1024)) % $(288)) >= $(32)) & (((r424_z / $(1024)) % $(288)) < $(64)) & ((((r424_z / $(1024)) % $(288)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_51_ptr) + ((((((r424_z / $(1024)) % $(288)) - $(32)) % $(32)) * $(1536)) + (((r424_z / $(128)) % $(8)) * $(192)) + (r424_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r424_value = r424_value.to(tl.bfloat16)
    r424_value = tl.broadcast_to(r424_value, [$(294912)])
    tl.store($(candidate_vslice_backward_54_ptr) + r424_z, r424_value)
    r425_z = tl.arange($(294912))
    r425_value = (tl.load($(candidate_vadd_44_ptr) + ((((r425_z / $(1024)) % $(288)) * $(1024)) + (((r425_z / $(128)) % $(8)) * $(128)) + (r425_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_54_ptr) + ((((r425_z / $(1024)) % $(288)) * $(1024)) + (((r425_z / $(128)) % $(8)) * $(128)) + (r425_z % $(128))), dtype=tl.bfloat16))
    r425_value = r425_value.to(tl.bfloat16)
    r425_value = tl.broadcast_to(r425_value, [$(294912)])
    tl.store($(candidate_vadd_49_ptr) + r425_z, r425_value)
    r426_z = tl.arange($(147456))
    r426_value = tl.where(((((r426_z / $(512)) % $(288)) >= $(32)) & (((r426_z / $(512)) % $(288)) < $(64)) & ((((r426_z / $(512)) % $(288)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_51_ptr) + ((((((r426_z / $(512)) % $(288)) - $(32)) % $(32)) * $(1536)) + (((r426_z / $(64)) % $(8)) * $(192)) + (r426_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r426_value = r426_value.to(tl.bfloat16)
    r426_value = tl.broadcast_to(r426_value, [$(147456)])
    tl.store($(candidate_vslice_backward_55_ptr) + r426_z, r426_value)
    r427_z = tl.arange($(147456))
    r427_value = (tl.load($(candidate_vadd_45_ptr) + ((((r427_z / $(512)) % $(288)) * $(512)) + (((r427_z / $(64)) % $(8)) * $(64)) + (r427_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_55_ptr) + ((((r427_z / $(512)) % $(288)) * $(512)) + (((r427_z / $(64)) % $(8)) * $(64)) + (r427_z % $(64))), dtype=tl.bfloat16))
    r427_value = r427_value.to(tl.bfloat16)
    r427_value = tl.broadcast_to(r427_value, [$(147456)])
    tl.store($(candidate_vadd_50_ptr) + r427_z, r427_value)
    r428_row = tl.arange($(32))
    r428_col = tl.arange($(128))
    r428_lane = tl.arange($(32))
    for r428_group in range($(8)) {
      r428_left = tl.load($(candidate_vto_copy_24_ptr) + (((((((r428_group % $(8)) * $(1024)) + (r428_lane[None, :] * $(32)) + r428_row[:, None]) / $(1024)) % $(8)) * $(1024)) + ((((((r428_group % $(8)) * $(1024)) + (r428_lane[None, :] * $(32)) + r428_row[:, None]) / $(32)) % $(32)) * $(32)) + ((((r428_group % $(8)) * $(1024)) + (r428_lane[None, :] * $(32)) + r428_row[:, None]) % $(32))), dtype=tl.bfloat16)
      r428_right = tl.load($(candidate_vmm_4_ptr) + (((((((r428_group % $(8)) * $(4096)) + (r428_lane[:, None] * $(128)) + r428_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r428_group % $(8)) * $(4096)) + (r428_lane[:, None] * $(128)) + r428_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r428_group % $(8)) * $(4096)) + (r428_lane[:, None] * $(128)) + r428_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r428_value = tl.dot(r428_left, r428_right)
      r428_value = r428_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_84_ptr) + ((r428_group * $(4096)) + (r428_row[:, None] * $(128)) + r428_col[None, :]), r428_value)
    }
    r429_row = tl.arange($(32))
    r429_col = tl.arange($(32))
    r429_lane = tl.arange($(128))
    for r429_group in range($(8)) {
      r429_left = tl.load($(candidate_vmm_4_ptr) + (((((((r429_group % $(8)) * $(4096)) + (r429_row[:, None] * $(128)) + r429_lane[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r429_group % $(8)) * $(4096)) + (r429_row[:, None] * $(128)) + r429_lane[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r429_group % $(8)) * $(4096)) + (r429_row[:, None] * $(128)) + r429_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r429_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r429_group % $(8)) * $(4096)) + (r429_col[None, :] * $(128)) + r429_lane[:, None]) / $(128)) % $(32)) * $(1024)) + ((((((r429_group % $(8)) * $(4096)) + (r429_col[None, :] * $(128)) + r429_lane[:, None]) / $(4096)) % $(8)) * $(128)) + ((((r429_group % $(8)) * $(4096)) + (r429_col[None, :] * $(128)) + r429_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r429_value = tl.dot(r429_left, r429_right)
      r429_value = r429_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_85_ptr) + ((r429_group * $(1024)) + (r429_row[:, None] * $(32)) + r429_col[None, :]), r429_value)
    }
    r430_z = tl.arange($(8192))
    r430_value = tl.load($(candidate_vbmm_85_ptr) + ((((r430_z / $(1024)) % $(8)) * $(1024)) + (((r430_z / $(32)) % $(32)) * $(32)) + (r430_z % $(32))), dtype=tl.bfloat16)
    r430_value = r430_value.to(tl.float32)
    r430_value = tl.broadcast_to(r430_value, [$(8192)])
    tl.store($(candidate_vto_copy_73_ptr) + r430_z, r430_value)
    r431_z = tl.arange($(8192))
    r431_lane = tl.arange($(32))
    r431_row_values = tl.load($(candidate_vsoftmax_ptr) + ((((r431_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r431_z[:, None] / $(32)) % $(32)) * $(32)) + (r431_lane[None, :] % $(32))), dtype=tl.float32)
    r431_row_grad = tl.load($(candidate_vto_copy_73_ptr) + ((((r431_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r431_z[:, None] / $(32)) % $(32)) * $(32)) + (r431_lane[None, :] % $(32))), dtype=tl.float32)
    r431_weighted = r431_row_grad * r431_row_values
    r431_dot_grad = tl.sum(r431_weighted, axis=1)
    r431_value = (tl.load($(candidate_vto_copy_73_ptr) + ((((r431_z / $(1024)) % $(8)) * $(1024)) + (((r431_z / $(32)) % $(32)) * $(32)) + (r431_z % $(32))), dtype=tl.float32) - r431_dot_grad) * tl.load($(candidate_vsoftmax_ptr) + ((((r431_z / $(1024)) % $(8)) * $(1024)) + (((r431_z / $(32)) % $(32)) * $(32)) + (r431_z % $(32))), dtype=tl.float32)
    r431_value = r431_value.to(tl.float32)
    r431_value = tl.broadcast_to(r431_value, [$(8192)])
    tl.store($(candidate_vsoftmax_backward_data_8_ptr) + r431_z, r431_value)
    r432_z = tl.arange($(8192))
    r432_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r432_z / $(32)) % $(32)) * $(32)) + (r432_z % $(32))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_8_ptr) + ((((r432_z / $(1024)) % $(8)) * $(1024)) + (((r432_z / $(32)) % $(32)) * $(32)) + (r432_z % $(32))), dtype=tl.float32))
    r432_value = r432_value.to(tl.float32)
    r432_value = tl.broadcast_to(r432_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_17_ptr) + r432_z, r432_value)
    r433_z = tl.arange($(8192))
    r433_value = (tl.load($(candidate_vmasked_fill_17_ptr) + ((((r433_z / $(1024)) % $(8)) * $(1024)) + (((r433_z / $(32)) % $(32)) * $(32)) + (r433_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r433_value = r433_value.to(tl.float32)
    r433_value = tl.broadcast_to(r433_value, [$(8192)])
    tl.store($(candidate_vmul_38_ptr) + r433_z, r433_value)
    r434_row = tl.arange($(192))
    r434_col = tl.arange($(32))
    r434_lane = tl.arange($(32))
    for r434_group in range($(8)) {
      r434_left = tl.load($(candidate_vto_copy_22_ptr) + (((((((r434_group % $(8)) * $(6144)) + (r434_lane[None, :] * $(192)) + r434_row[:, None]) / $(6144)) % $(8)) * $(6144)) + ((((((r434_group % $(8)) * $(6144)) + (r434_lane[None, :] * $(192)) + r434_row[:, None]) / $(192)) % $(32)) * $(192)) + ((((r434_group % $(8)) * $(6144)) + (r434_lane[None, :] * $(192)) + r434_row[:, None]) % $(192))), dtype=tl.float32)
      r434_right = tl.load($(candidate_vmul_38_ptr) + (((((((r434_group % $(8)) * $(1024)) + (r434_lane[:, None] * $(32)) + r434_col[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r434_group % $(8)) * $(1024)) + (r434_lane[:, None] * $(32)) + r434_col[None, :]) / $(32)) % $(32)) * $(32)) + ((((r434_group % $(8)) * $(1024)) + (r434_lane[:, None] * $(32)) + r434_col[None, :]) % $(32))), dtype=tl.float32)
      r434_value = tl.dot(r434_left, r434_right)
      r434_value = r434_value.to(tl.float32)
      tl.store($(candidate_vbmm_86_ptr) + ((r434_group * $(6144)) + (r434_row[:, None] * $(32)) + r434_col[None, :]), r434_value)
    }
    r435_row = tl.arange($(32))
    r435_col = tl.arange($(192))
    r435_lane = tl.arange($(32))
    for r435_group in range($(8)) {
      r435_left = tl.load($(candidate_vmul_38_ptr) + (((((((r435_group % $(8)) * $(1024)) + (r435_row[:, None] * $(32)) + r435_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r435_group % $(8)) * $(1024)) + (r435_row[:, None] * $(32)) + r435_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r435_group % $(8)) * $(1024)) + (r435_row[:, None] * $(32)) + r435_lane[None, :]) % $(32))), dtype=tl.float32)
      r435_right = tl.load($(candidate_vto_copy_23_ptr) + (((((((r435_group % $(8)) * $(6144)) + (r435_col[None, :] * $(32)) + r435_lane[:, None]) / $(6144)) % $(8)) * $(6144)) + (((((r435_group % $(8)) * $(6144)) + (r435_col[None, :] * $(32)) + r435_lane[:, None]) % $(32)) * $(192)) + (((((r435_group % $(8)) * $(6144)) + (r435_col[None, :] * $(32)) + r435_lane[:, None]) / $(32)) % $(192))), dtype=tl.float32)
      r435_value = tl.dot(r435_left, r435_right)
      r435_value = r435_value.to(tl.float32)
      tl.store($(candidate_vbmm_87_ptr) + ((r435_group * $(6144)) + (r435_row[:, None] * $(192)) + r435_col[None, :]), r435_value)
    }
    r436_z = tl.arange($(98304))
    r436_value = tl.where(((((r436_z / $(1024)) % $(96)) >= $(0)) & (((r436_z / $(1024)) % $(96)) < $(32)) & ((((r436_z / $(1024)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_84_ptr) + ((((((((r436_z / $(128)) % $(8)) * $(4096)) + ((((r436_z / $(1024)) % $(96)) - $(0)) * $(128)) + (r436_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r436_z / $(128)) % $(8)) * $(32)) + (((r436_z / $(1024)) % $(96)) - $(0))) % $(32)) * $(128)) + (r436_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r436_value = r436_value.to(tl.bfloat16)
    r436_value = tl.broadcast_to(r436_value, [$(98304)])
    tl.store($(candidate_vslice_backward_56_ptr) + r436_z, r436_value)
    r437_z = tl.arange($(49152))
    r437_value = tl.load($(candidate_vbmm_86_ptr) + ((((r437_z / $(6144)) % $(8)) * $(6144)) + ((r437_z % $(192)) * $(32)) + ((r437_z / $(192)) % $(32))), dtype=tl.float32)
    r437_value = r437_value.to(tl.bfloat16)
    r437_value = tl.broadcast_to(r437_value, [$(49152)])
    tl.store($(candidate_vto_copy_74_ptr) + r437_z, r437_value)
    r438_z = tl.arange($(147456))
    r438_value = tl.where(((((r438_z / $(1536)) % $(96)) >= $(0)) & (((r438_z / $(1536)) % $(96)) < $(32)) & ((((r438_z / $(1536)) % $(96)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_74_ptr) + ((((r438_z / $(192)) % $(8)) * $(6144)) + (((((r438_z / $(1536)) % $(96)) - $(0)) % $(32)) * $(192)) + (r438_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r438_value = r438_value.to(tl.bfloat16)
    r438_value = tl.broadcast_to(r438_value, [$(147456)])
    tl.store($(candidate_vslice_backward_57_ptr) + r438_z, r438_value)
    r439_z = tl.arange($(49152))
    r439_value = tl.load($(candidate_vbmm_87_ptr) + ((((r439_z / $(6144)) % $(8)) * $(6144)) + (((r439_z / $(192)) % $(32)) * $(192)) + (r439_z % $(192))), dtype=tl.float32)
    r439_value = r439_value.to(tl.bfloat16)
    r439_value = tl.broadcast_to(r439_value, [$(49152)])
    tl.store($(candidate_vto_copy_75_ptr) + r439_z, r439_value)
    r440_z = tl.arange($(49152))
    r440_value = tl.where(((((r440_z / $(1536)) % $(32)) >= $(0)) & (((r440_z / $(1536)) % $(32)) < $(32)) & ((((r440_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_75_ptr) + ((((r440_z / $(192)) % $(8)) * $(6144)) + (((((r440_z / $(1536)) % $(32)) - $(0)) % $(32)) * $(192)) + (r440_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r440_value = r440_value.to(tl.bfloat16)
    r440_value = tl.broadcast_to(r440_value, [$(49152)])
    tl.store($(candidate_vslice_backward_58_ptr) + r440_z, r440_value)
    r441_z = tl.arange($(65536))
    r441_value = tl.cast(0.0, tl.bfloat16)
    r441_value = r441_value.to(tl.bfloat16)
    r441_value = tl.broadcast_to(r441_value, [$(65536)])
    tl.store($(candidate_vfill_16_ptr) + r441_z, r441_value)
    r442_z = tl.arange($(98304))
    r442_value = tl.where(((((r442_z / $(1024)) % $(96)) >= $(32)) & (((r442_z / $(1024)) % $(96)) < $(96)) & ((((r442_z / $(1024)) % $(96)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vfill_16_ptr) + ((((((r442_z / $(1024)) % $(96)) - $(32)) % $(64)) * $(1024)) + (((r442_z / $(128)) % $(8)) * $(128)) + (r442_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_56_ptr) + ((((r442_z / $(1024)) % $(96)) * $(1024)) + (((r442_z / $(128)) % $(8)) * $(128)) + (r442_z % $(128))), dtype=tl.bfloat16))
    r442_value = r442_value.to(tl.bfloat16)
    r442_value = tl.broadcast_to(r442_value, [$(98304)])
    tl.store($(candidate_vslice_scatter_16_ptr) + r442_z, r442_value)
    r443_z = tl.arange($(98304))
    r443_value = tl.cast(0.0, tl.bfloat16)
    r443_value = r443_value.to(tl.bfloat16)
    r443_value = tl.broadcast_to(r443_value, [$(98304)])
    tl.store($(candidate_vfill_17_ptr) + r443_z, r443_value)
    r444_z = tl.arange($(147456))
    r444_value = tl.where(((((r444_z / $(1536)) % $(96)) >= $(32)) & (((r444_z / $(1536)) % $(96)) < $(96)) & ((((r444_z / $(1536)) % $(96)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vfill_17_ptr) + ((((((r444_z / $(1536)) % $(96)) - $(32)) % $(64)) * $(1536)) + (((r444_z / $(192)) % $(8)) * $(192)) + (r444_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_backward_57_ptr) + ((((r444_z / $(1536)) % $(96)) * $(1536)) + (((r444_z / $(192)) % $(8)) * $(192)) + (r444_z % $(192))), dtype=tl.bfloat16))
    r444_value = r444_value.to(tl.bfloat16)
    r444_value = tl.broadcast_to(r444_value, [$(147456)])
    tl.store($(candidate_vslice_scatter_17_ptr) + r444_z, r444_value)
    r445_z = tl.arange($(6144))
    r445_lane = tl.arange($(8))
    r445_value = tl.sum(tl.load($(candidate_vslice_scatter_17_ptr) + ((((r445_z[:, None] / $(64)) % $(96)) * $(1536)) + (r445_lane[None, :] * $(192)) + (r445_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r445_value = r445_value.to(tl.bfloat16)
    r445_value = tl.broadcast_to(r445_value, [$(6144)])
    tl.store($(candidate_vsum_9_ptr) + r445_z, r445_value)
    r446_row = tl.arange($(96))
    r446_col = tl.arange($(512))
    r446_lane = tl.arange($(1024))
    for r446_group in range($(1)) {
      r446_left = tl.load($(candidate_vslice_scatter_16_ptr) + ((((((r446_row[:, None] * $(1024)) + r446_lane[None, :]) / $(1024)) % $(96)) * $(1024)) + (((((r446_row[:, None] * $(1024)) + r446_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r446_row[:, None] * $(1024)) + r446_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r446_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r446_col[None, :] * $(1024)) + r446_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r446_col[None, :] * $(1024)) + r446_lane[:, None]) % $(128)) * $(512)) + ((((r446_col[None, :] * $(1024)) + r446_lane[:, None]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r446_value = tl.dot(r446_left, r446_right)
      r446_value = r446_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_88_ptr) + ((r446_group * $(49152)) + (r446_row[:, None] * $(512)) + r446_col[None, :]), r446_value)
    }
    r447_row = tl.arange($(96))
    r447_col = tl.arange($(512))
    r447_lane = tl.arange($(1024))
    for r447_group in range($(1)) {
      r447_left = tl.load($(candidate_vslice_scatter_17_ptr) + ((((((r447_row[:, None] * $(1024)) + r447_lane[None, :]) / $(1024)) % $(96)) * $(1536)) + (((((r447_row[:, None] * $(1024)) + r447_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r447_row[:, None] * $(1024)) + r447_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r447_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r447_col[None, :] * $(1024)) + r447_lane[:, None]) / $(128)) % $(8)) * $(131072)) + ((((r447_col[None, :] * $(1024)) + r447_lane[:, None]) % $(128)) * $(512)) + ((((r447_col[None, :] * $(1024)) + r447_lane[:, None]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r447_value = tl.dot(r447_left, r447_right)
      r447_value = r447_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_89_ptr) + ((r447_group * $(49152)) + (r447_row[:, None] * $(512)) + r447_col[None, :]), r447_value)
    }
    r448_z = tl.arange($(49152))
    r448_value = (tl.load($(candidate_vbmm_88_ptr) + ((((r448_z / $(512)) % $(96)) * $(512)) + (r448_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_89_ptr) + ((((r448_z / $(512)) % $(96)) * $(512)) + (r448_z % $(512))), dtype=tl.bfloat16))
    r448_value = r448_value.to(tl.bfloat16)
    r448_value = tl.broadcast_to(r448_value, [$(49152)])
    tl.store($(candidate_vadd_51_ptr) + r448_z, r448_value)
    r449_z = tl.arange($(49152))
    r449_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_51_ptr) + ((((r449_z / $(512)) % $(96)) * $(512)) + (r449_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r449_value = r449_value.to(tl.bfloat16)
    r449_value = tl.broadcast_to(r449_value, [$(49152)])
    tl.store($(candidate_vselect_backward_8_ptr) + r449_z, r449_value)
    r450_z = tl.arange($(147456))
    r450_value = tl.where(((((r450_z / $(512)) % $(288)) >= $(0)) & (((r450_z / $(512)) % $(288)) < $(96)) & ((((r450_z / $(512)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_8_ptr) + ((((((r450_z / $(512)) % $(288)) - $(0)) % $(96)) * $(512)) + (r450_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r450_value = r450_value.to(tl.bfloat16)
    r450_value = tl.broadcast_to(r450_value, [$(147456)])
    tl.store($(candidate_vslice_backward_59_ptr) + r450_z, r450_value)
    r451_z = tl.arange($(147456))
    r451_value = (tl.load($(candidate_vadd_47_ptr) + ((((r451_z / $(512)) % $(288)) * $(512)) + (r451_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_59_ptr) + ((((r451_z / $(512)) % $(288)) * $(512)) + (r451_z % $(512))), dtype=tl.bfloat16))
    r451_value = r451_value.to(tl.bfloat16)
    r451_value = tl.broadcast_to(r451_value, [$(147456)])
    tl.store($(candidate_vadd_52_ptr) + r451_z, r451_value)
    r452_z = tl.arange($(18432))
    r452_value = tl.where(((((r452_z / $(64)) % $(288)) >= $(0)) & (((r452_z / $(64)) % $(288)) < $(96)) & ((((r452_z / $(64)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_9_ptr) + ((((((r452_z / $(64)) % $(288)) - $(0)) % $(96)) * $(64)) + (r452_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r452_value = r452_value.to(tl.bfloat16)
    r452_value = tl.broadcast_to(r452_value, [$(18432)])
    tl.store($(candidate_vslice_backward_60_ptr) + r452_z, r452_value)
    r453_z = tl.arange($(18432))
    r453_value = (tl.load($(candidate_vadd_48_ptr) + ((((r453_z / $(64)) % $(288)) * $(64)) + (r453_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_60_ptr) + ((((r453_z / $(64)) % $(288)) * $(64)) + (r453_z % $(64))), dtype=tl.bfloat16))
    r453_value = r453_value.to(tl.bfloat16)
    r453_value = tl.broadcast_to(r453_value, [$(18432)])
    tl.store($(candidate_vadd_53_ptr) + r453_z, r453_value)
    r454_z = tl.arange($(294912))
    r454_value = tl.where(((((r454_z / $(1024)) % $(288)) >= $(0)) & (((r454_z / $(1024)) % $(288)) < $(32)) & ((((r454_z / $(1024)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_58_ptr) + ((((((r454_z / $(1024)) % $(288)) - $(0)) % $(32)) * $(1536)) + (((r454_z / $(128)) % $(8)) * $(192)) + (r454_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r454_value = r454_value.to(tl.bfloat16)
    r454_value = tl.broadcast_to(r454_value, [$(294912)])
    tl.store($(candidate_vslice_backward_61_ptr) + r454_z, r454_value)
    r455_z = tl.arange($(294912))
    r455_value = (tl.load($(candidate_vadd_49_ptr) + ((((r455_z / $(1024)) % $(288)) * $(1024)) + (((r455_z / $(128)) % $(8)) * $(128)) + (r455_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_61_ptr) + ((((r455_z / $(1024)) % $(288)) * $(1024)) + (((r455_z / $(128)) % $(8)) * $(128)) + (r455_z % $(128))), dtype=tl.bfloat16))
    r455_value = r455_value.to(tl.bfloat16)
    r455_value = tl.broadcast_to(r455_value, [$(294912)])
    tl.store($(candidate_vadd_54_ptr) + r455_z, r455_value)
    r456_z = tl.arange($(147456))
    r456_value = tl.where(((((r456_z / $(512)) % $(288)) >= $(0)) & (((r456_z / $(512)) % $(288)) < $(32)) & ((((r456_z / $(512)) % $(288)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vslice_backward_58_ptr) + ((((((r456_z / $(512)) % $(288)) - $(0)) % $(32)) * $(1536)) + (((r456_z / $(64)) % $(8)) * $(192)) + (r456_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r456_value = r456_value.to(tl.bfloat16)
    r456_value = tl.broadcast_to(r456_value, [$(147456)])
    tl.store($(candidate_vslice_backward_62_ptr) + r456_z, r456_value)
    r457_z = tl.arange($(147456))
    r457_value = (tl.load($(candidate_vadd_50_ptr) + ((((r457_z / $(512)) % $(288)) * $(512)) + (((r457_z / $(64)) % $(8)) * $(64)) + (r457_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_62_ptr) + ((((r457_z / $(512)) % $(288)) * $(512)) + (((r457_z / $(64)) % $(8)) * $(64)) + (r457_z % $(64))), dtype=tl.bfloat16))
    r457_value = r457_value.to(tl.bfloat16)
    r457_value = tl.broadcast_to(r457_value, [$(147456)])
    tl.store($(candidate_vadd_55_ptr) + r457_z, r457_value)
    r458_z = tl.arange($(165888))
    r458_value = tl.where((r458_z % $(576)) < $(512), tl.load($(candidate_vadd_52_ptr) + ((((r458_z / $(576)) % $(288)) * $(512)) + (((r458_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_53_ptr) + ((((r458_z / $(576)) % $(288)) * $(64)) + (((r458_z % $(576)) - $(512)) % $(64))), dtype=tl.bfloat16))
    r458_value = r458_value.to(tl.bfloat16)
    r458_value = tl.broadcast_to(r458_value, [$(165888)])
    tl.store($(candidate_vcat_36_ptr) + r458_z, r458_value)
    r459_z = tl.arange($(9216))
    r459_value = (tl.load($(candidate_vcat_36_ptr) + ((((r459_z / $(32)) % $(288)) * $(576)) + (r459_z % $(32)) + $(544)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r459_z / $(32)) % $(288)) * $(64)) + (r459_z % $(32))), dtype=tl.bfloat16))
    r459_value = r459_value.to(tl.bfloat16)
    r459_value = tl.broadcast_to(r459_value, [$(9216)])
    tl.store($(candidate_vmul_39_ptr) + r459_z, r459_value)
    r460_z = tl.arange($(9216))
    r460_value = (tl.load($(candidate_vcat_36_ptr) + ((((r460_z / $(32)) % $(288)) * $(576)) + (r460_z % $(32)) + $(544)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r460_z / $(32)) % $(288)) * $(64)) + (r460_z % $(32))), dtype=tl.bfloat16))
    r460_value = r460_value.to(tl.bfloat16)
    r460_value = tl.broadcast_to(r460_value, [$(9216)])
    tl.store($(candidate_vmul_40_ptr) + r460_z, r460_value)
    r461_z = tl.arange($(9216))
    r461_value = -tl.load($(candidate_vcat_36_ptr) + ((((r461_z / $(32)) % $(288)) * $(576)) + (r461_z % $(32)) + $(512)), dtype=tl.bfloat16)
    r461_value = r461_value.to(tl.bfloat16)
    r461_value = tl.broadcast_to(r461_value, [$(9216)])
    tl.store($(candidate_vneg_ptr) + r461_z, r461_value)
    r462_z = tl.arange($(9216))
    r462_value = (tl.load($(candidate_vneg_ptr) + ((((r462_z / $(32)) % $(288)) * $(32)) + (r462_z % $(32))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r462_z / $(32)) % $(288)) * $(64)) + (r462_z % $(32))), dtype=tl.bfloat16))
    r462_value = r462_value.to(tl.bfloat16)
    r462_value = tl.broadcast_to(r462_value, [$(9216)])
    tl.store($(candidate_vmul_41_ptr) + r462_z, r462_value)
    r463_z = tl.arange($(9216))
    r463_value = (tl.load($(candidate_vmul_40_ptr) + ((((r463_z / $(32)) % $(288)) * $(32)) + (r463_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_41_ptr) + ((((r463_z / $(32)) % $(288)) * $(32)) + (r463_z % $(32))), dtype=tl.bfloat16))
    r463_value = r463_value.to(tl.bfloat16)
    r463_value = tl.broadcast_to(r463_value, [$(9216)])
    tl.store($(candidate_vadd_56_ptr) + r463_z, r463_value)
    r464_z = tl.arange($(9216))
    r464_value = (tl.load($(candidate_vcat_36_ptr) + ((((r464_z / $(32)) % $(288)) * $(576)) + (r464_z % $(32)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r464_z / $(32)) % $(288)) * $(64)) + (r464_z % $(32))), dtype=tl.bfloat16))
    r464_value = r464_value.to(tl.bfloat16)
    r464_value = tl.broadcast_to(r464_value, [$(9216)])
    tl.store($(candidate_vmul_42_ptr) + r464_z, r464_value)
    r465_z = tl.arange($(9216))
    r465_value = (tl.load($(candidate_vmul_39_ptr) + ((((r465_z / $(32)) % $(288)) * $(32)) + (r465_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_42_ptr) + ((((r465_z / $(32)) % $(288)) * $(32)) + (r465_z % $(32))), dtype=tl.bfloat16))
    r465_value = r465_value.to(tl.bfloat16)
    r465_value = tl.broadcast_to(r465_value, [$(9216)])
    tl.store($(candidate_vadd_57_ptr) + r465_z, r465_value)
    r466_z = tl.arange($(73728))
    r466_value = (tl.load($(candidate_vadd_55_ptr) + ((((r466_z / $(32)) % $(288)) * $(512)) + (((r466_z / $(9216)) % $(8)) * $(64)) + (r466_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r466_z / $(32)) % $(288)) * $(64)) + (r466_z % $(32))), dtype=tl.bfloat16))
    r466_value = r466_value.to(tl.bfloat16)
    r466_value = tl.broadcast_to(r466_value, [$(73728)])
    tl.store($(candidate_vmul_43_ptr) + r466_z, r466_value)
    r467_z = tl.arange($(73728))
    r467_value = (tl.load($(candidate_vadd_55_ptr) + ((((r467_z / $(32)) % $(288)) * $(512)) + (((r467_z / $(9216)) % $(8)) * $(64)) + (r467_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r467_z / $(32)) % $(288)) * $(64)) + (r467_z % $(32))), dtype=tl.bfloat16))
    r467_value = r467_value.to(tl.bfloat16)
    r467_value = tl.broadcast_to(r467_value, [$(73728)])
    tl.store($(candidate_vmul_44_ptr) + r467_z, r467_value)
    r468_z = tl.arange($(73728))
    r468_value = -tl.load($(candidate_vadd_55_ptr) + ((((r468_z / $(32)) % $(288)) * $(512)) + (((r468_z / $(9216)) % $(8)) * $(64)) + (r468_z % $(32))), dtype=tl.bfloat16)
    r468_value = r468_value.to(tl.bfloat16)
    r468_value = tl.broadcast_to(r468_value, [$(73728)])
    tl.store($(candidate_vneg_1_ptr) + r468_z, r468_value)
    r469_z = tl.arange($(73728))
    r469_value = (tl.load($(candidate_vneg_1_ptr) + ((((r469_z / $(9216)) % $(8)) * $(9216)) + (((r469_z / $(32)) % $(288)) * $(32)) + (r469_z % $(32))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r469_z / $(32)) % $(288)) * $(64)) + (r469_z % $(32))), dtype=tl.bfloat16))
    r469_value = r469_value.to(tl.bfloat16)
    r469_value = tl.broadcast_to(r469_value, [$(73728)])
    tl.store($(candidate_vmul_45_ptr) + r469_z, r469_value)
    r470_z = tl.arange($(73728))
    r470_value = (tl.load($(candidate_vmul_44_ptr) + ((((r470_z / $(9216)) % $(8)) * $(9216)) + (((r470_z / $(32)) % $(288)) * $(32)) + (r470_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_45_ptr) + ((((r470_z / $(9216)) % $(8)) * $(9216)) + (((r470_z / $(32)) % $(288)) * $(32)) + (r470_z % $(32))), dtype=tl.bfloat16))
    r470_value = r470_value.to(tl.bfloat16)
    r470_value = tl.broadcast_to(r470_value, [$(73728)])
    tl.store($(candidate_vadd_58_ptr) + r470_z, r470_value)
    r471_z = tl.arange($(73728))
    r471_value = (tl.load($(candidate_vadd_55_ptr) + ((((r471_z / $(32)) % $(288)) * $(512)) + (((r471_z / $(9216)) % $(8)) * $(64)) + (r471_z % $(32))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r471_z / $(32)) % $(288)) * $(64)) + (r471_z % $(32))), dtype=tl.bfloat16))
    r471_value = r471_value.to(tl.bfloat16)
    r471_value = tl.broadcast_to(r471_value, [$(73728)])
    tl.store($(candidate_vmul_46_ptr) + r471_z, r471_value)
    r472_z = tl.arange($(73728))
    r472_value = (tl.load($(candidate_vmul_43_ptr) + ((((r472_z / $(9216)) % $(8)) * $(9216)) + (((r472_z / $(32)) % $(288)) * $(32)) + (r472_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_46_ptr) + ((((r472_z / $(9216)) % $(8)) * $(9216)) + (((r472_z / $(32)) % $(288)) * $(32)) + (r472_z % $(32))), dtype=tl.bfloat16))
    r472_value = r472_value.to(tl.bfloat16)
    r472_value = tl.broadcast_to(r472_value, [$(73728)])
    tl.store($(candidate_vadd_59_ptr) + r472_z, r472_value)
    r473_z = tl.arange($(18432))
    r473_value = tl.where((((r473_z % $(64)) >= $(1)) & ((r473_z % $(64)) < $(64)) & (((r473_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_56_ptr) + ((((r473_z / $(64)) % $(288)) * $(32)) + ((((r473_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r473_value = r473_value.to(tl.bfloat16)
    r473_value = tl.broadcast_to(r473_value, [$(18432)])
    tl.store($(candidate_vslice_backward_63_ptr) + r473_z, r473_value)
    r474_z = tl.arange($(18432))
    r474_value = tl.where((((r474_z % $(64)) >= $(0)) & ((r474_z % $(64)) < $(64)) & (((r474_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_57_ptr) + ((((r474_z / $(64)) % $(288)) * $(32)) + ((((r474_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r474_value = r474_value.to(tl.bfloat16)
    r474_value = tl.broadcast_to(r474_value, [$(18432)])
    tl.store($(candidate_vslice_backward_64_ptr) + r474_z, r474_value)
    r475_z = tl.arange($(18432))
    r475_value = (tl.load($(candidate_vslice_backward_63_ptr) + ((((r475_z / $(64)) % $(288)) * $(64)) + (r475_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_64_ptr) + ((((r475_z / $(64)) % $(288)) * $(64)) + (r475_z % $(64))), dtype=tl.bfloat16))
    r475_value = r475_value.to(tl.bfloat16)
    r475_value = tl.broadcast_to(r475_value, [$(18432)])
    tl.store($(candidate_vadd_60_ptr) + r475_z, r475_value)
    r476_z = tl.arange($(147456))
    r476_value = tl.where((((r476_z % $(64)) >= $(1)) & ((r476_z % $(64)) < $(64)) & (((r476_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_58_ptr) + ((((r476_z / $(18432)) % $(8)) * $(9216)) + (((r476_z / $(64)) % $(288)) * $(32)) + ((((r476_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r476_value = r476_value.to(tl.bfloat16)
    r476_value = tl.broadcast_to(r476_value, [$(147456)])
    tl.store($(candidate_vslice_backward_65_ptr) + r476_z, r476_value)
    r477_z = tl.arange($(147456))
    r477_value = tl.where((((r477_z % $(64)) >= $(0)) & ((r477_z % $(64)) < $(64)) & (((r477_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_59_ptr) + ((((r477_z / $(18432)) % $(8)) * $(9216)) + (((r477_z / $(64)) % $(288)) * $(32)) + ((((r477_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r477_value = r477_value.to(tl.bfloat16)
    r477_value = tl.broadcast_to(r477_value, [$(147456)])
    tl.store($(candidate_vslice_backward_66_ptr) + r477_z, r477_value)
    r478_z = tl.arange($(147456))
    r478_value = (tl.load($(candidate_vslice_backward_65_ptr) + ((((r478_z / $(18432)) % $(8)) * $(18432)) + (((r478_z / $(64)) % $(288)) * $(64)) + (r478_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_66_ptr) + ((((r478_z / $(18432)) % $(8)) * $(18432)) + (((r478_z / $(64)) % $(288)) * $(64)) + (r478_z % $(64))), dtype=tl.bfloat16))
    r478_value = r478_value.to(tl.bfloat16)
    r478_value = tl.broadcast_to(r478_value, [$(147456)])
    tl.store($(candidate_vadd_61_ptr) + r478_z, r478_value)
    r479_z = tl.arange($(442368))
    r479_value = tl.where((r479_z % $(192)) < $(128), tl.load($(candidate_vadd_54_ptr) + ((((((((r479_z / $(192)) % $(288)) * $(1024)) + (((r479_z / $(55296)) % $(8)) * $(128)) + ((r479_z % $(192)) - $(0))) / $(1024)) % $(288)) * $(1024)) + (((((((r479_z / $(192)) % $(288)) * $(1024)) + (((r479_z / $(55296)) % $(8)) * $(128)) + ((r479_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r479_z / $(192)) % $(288)) * $(1024)) + (((r479_z / $(55296)) % $(8)) * $(128)) + ((r479_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_61_ptr) + ((((r479_z / $(55296)) % $(8)) * $(18432)) + (((r479_z / $(192)) % $(288)) * $(64)) + (((r479_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r479_value = r479_value.to(tl.bfloat16)
    r479_value = tl.broadcast_to(r479_value, [$(442368)])
    tl.store($(candidate_vcat_37_ptr) + r479_z, r479_value)
    r480_z = tl.arange($(147456))
    r480_value = tl.load($(candidate_vcat_36_ptr) + ((((r480_z / $(512)) % $(288)) * $(576)) + (r480_z % $(512))), dtype=tl.bfloat16)
    r480_value = r480_value.to(tl.float32)
    r480_value = tl.broadcast_to(r480_value, [$(147456)])
    tl.store($(candidate_vto_copy_76_ptr) + r480_z, r480_value)
    r481_z = tl.arange($(147456))
    r481_value = (tl.load($(candidate_vto_copy_76_ptr) + ((((r481_z / $(512)) % $(288)) * $(512)) + (r481_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r481_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r481_value = r481_value.to(tl.float32)
    r481_value = tl.broadcast_to(r481_value, [$(147456)])
    tl.store($(candidate_vmul_47_ptr) + r481_z, r481_value)
    r482_z = tl.arange($(147456))
    r482_value = (tl.load($(candidate_vmul_47_ptr) + ((((r482_z / $(512)) % $(288)) * $(512)) + (r482_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r482_z / $(512)) % $(288)) * $(512)) + (r482_z % $(512))), dtype=tl.float32))
    r482_value = r482_value.to(tl.float32)
    r482_value = tl.broadcast_to(r482_value, [$(147456)])
    tl.store($(candidate_vmul_48_ptr) + r482_z, r482_value)
    r483_z = tl.arange($(147456))
    r483_value = (tl.load($(candidate_vmul_47_ptr) + ((((r483_z / $(512)) % $(288)) * $(512)) + (r483_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r483_z / $(512)) % $(288)), dtype=tl.float32))
    r483_value = r483_value.to(tl.float32)
    r483_value = tl.broadcast_to(r483_value, [$(147456)])
    tl.store($(candidate_vmul_49_ptr) + r483_z, r483_value)
    r484_z = tl.arange($(288))
    r484_lane = tl.arange($(512))
    r484_value = tl.sum(tl.load($(candidate_vmul_48_ptr) + ((r484_z[:, None] * $(512)) + r484_lane[None, :]), dtype=tl.float32), axis=1)
    r484_value = r484_value.to(tl.float32)
    r484_value = tl.broadcast_to(r484_value, [$(288)])
    tl.store($(candidate_vsum_10_ptr) + r484_z, r484_value)
    r485_z = tl.arange($(288))
    r485_value = ((tl.load($(candidate_vrsqrt_1_ptr) + r485_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r485_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r485_z, dtype=tl.float32)))
    r485_value = r485_value.to(tl.float32)
    r485_value = tl.broadcast_to(r485_value, [$(288)])
    tl.store($(candidate_vpow_4_ptr) + r485_z, r485_value)
    r486_z = tl.arange($(288))
    r486_value = (tl.load($(candidate_vsum_10_ptr) + r486_z, dtype=tl.float32) * -0.5)
    r486_value = r486_value.to(tl.float32)
    r486_value = tl.broadcast_to(r486_value, [$(288)])
    tl.store($(candidate_vmul_50_ptr) + r486_z, r486_value)
    r487_z = tl.arange($(288))
    r487_value = (tl.load($(candidate_vmul_50_ptr) + r487_z, dtype=tl.float32) * tl.load($(candidate_vpow_4_ptr) + r487_z, dtype=tl.float32))
    r487_value = r487_value.to(tl.float32)
    r487_value = tl.broadcast_to(r487_value, [$(288)])
    tl.store($(candidate_vmul_51_ptr) + r487_z, r487_value)
    r488_z = tl.arange($(147456))
    r488_value = (tl.load($(candidate_vmul_51_ptr) + ((r488_z / $(512)) % $(288)), dtype=tl.float32) / 512.0)
    r488_value = r488_value.to(tl.float32)
    r488_value = tl.broadcast_to(r488_value, [$(147456)])
    tl.store($(candidate_vdiv_1_ptr) + r488_z, r488_value)
    r489_z = tl.arange($(147456))
    r489_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r489_z / $(512)) % $(288)) * $(512)) + (r489_z % $(512))), dtype=tl.float32)))
    r489_value = r489_value.to(tl.float32)
    r489_value = tl.broadcast_to(r489_value, [$(147456)])
    tl.store($(candidate_vpow_5_ptr) + r489_z, r489_value)
    r490_z = tl.arange($(147456))
    r490_value = (tl.load($(candidate_vpow_5_ptr) + ((((r490_z / $(512)) % $(288)) * $(512)) + (r490_z % $(512))), dtype=tl.float32) * 2.0)
    r490_value = r490_value.to(tl.float32)
    r490_value = tl.broadcast_to(r490_value, [$(147456)])
    tl.store($(candidate_vmul_52_ptr) + r490_z, r490_value)
    r491_z = tl.arange($(147456))
    r491_value = (tl.load($(candidate_vdiv_1_ptr) + ((((r491_z / $(512)) % $(288)) * $(512)) + (r491_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vmul_52_ptr) + ((((r491_z / $(512)) % $(288)) * $(512)) + (r491_z % $(512))), dtype=tl.float32))
    r491_value = r491_value.to(tl.float32)
    r491_value = tl.broadcast_to(r491_value, [$(147456)])
    tl.store($(candidate_vmul_53_ptr) + r491_z, r491_value)
    r492_z = tl.arange($(147456))
    r492_value = (tl.load($(candidate_vmul_49_ptr) + ((((r492_z / $(512)) % $(288)) * $(512)) + (r492_z % $(512))), dtype=tl.float32) + tl.load($(candidate_vmul_53_ptr) + ((((r492_z / $(512)) % $(288)) * $(512)) + (r492_z % $(512))), dtype=tl.float32))
    r492_value = r492_value.to(tl.float32)
    r492_value = tl.broadcast_to(r492_value, [$(147456)])
    tl.store($(candidate_vadd_62_ptr) + r492_z, r492_value)
    r493_z = tl.arange($(147456))
    r493_value = tl.load($(candidate_vadd_62_ptr) + ((((r493_z / $(512)) % $(288)) * $(512)) + (r493_z % $(512))), dtype=tl.float32)
    r493_value = r493_value.to(tl.bfloat16)
    r493_value = tl.broadcast_to(r493_value, [$(147456)])
    tl.store($(candidate_vto_copy_77_ptr) + r493_z, r493_value)
    r494_z = tl.arange($(165888))
    r494_value = tl.where((r494_z % $(576)) < $(512), tl.load($(candidate_vto_copy_77_ptr) + ((((r494_z / $(576)) % $(288)) * $(512)) + (((r494_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_60_ptr) + ((((((((r494_z / $(576)) % $(288)) * $(64)) + ((r494_z % $(576)) - $(512))) / $(64)) % $(288)) * $(64)) + (((((r494_z / $(576)) % $(288)) * $(64)) + ((r494_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r494_value = r494_value.to(tl.bfloat16)
    r494_value = tl.broadcast_to(r494_value, [$(165888)])
    tl.store($(candidate_vcat_38_ptr) + r494_z, r494_value)
    r495_row = tl.arange($(288))
    r495_col = tl.arange($(512))
    r495_lane = tl.arange($(576))
    r495_left = tl.load($(candidate_vcat_38_ptr) + ((((((r495_row[:, None] * $(576)) + r495_lane[None, :]) / $(576)) % $(288)) * $(576)) + (((r495_row[:, None] * $(576)) + r495_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r495_right = tl.load($(candidate_vto_copy_3_ptr) + (((r495_lane[:, None] % $(576)) * $(512)) + (r495_col[None, :] % $(512))), dtype=tl.bfloat16)
    r495_value = tl.dot(r495_left, r495_right)
    r495_value = r495_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_5_ptr) + ((r495_row[:, None] * $(512)) + r495_col[None, :]), r495_value)
    r496_row = tl.arange($(288))
    r496_col = tl.arange($(256))
    r496_lane = tl.arange($(1536))
    r496_left = tl.load($(candidate_vcat_37_ptr) + ((((((((((r496_row[:, None] * $(1536)) + r496_lane[None, :]) / $(1536)) % $(288)) * $(1536)) + (((r496_row[:, None] * $(1536)) + r496_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(55296)) + (((((r496_row[:, None] * $(1536)) + r496_lane[None, :]) / $(1536)) % $(288)) * $(192)) + (((((((r496_row[:, None] * $(1536)) + r496_lane[None, :]) / $(1536)) % $(288)) * $(1536)) + (((r496_row[:, None] * $(1536)) + r496_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r496_right = tl.load($(candidate_vto_copy_2_ptr) + (((r496_lane[:, None] % $(1536)) * $(256)) + (r496_col[None, :] % $(256))), dtype=tl.bfloat16)
    r496_value = tl.dot(r496_left, r496_right)
    r496_value = r496_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r496_row[:, None] * $(256)) + r496_col[None, :]), r496_value)
    r497_z = tl.arange($(73728))
    r497_value = tl.load($(candidate_vmm_6_ptr) + ((((r497_z / $(256)) % $(288)) * $(256)) + (r497_z % $(256))), dtype=tl.bfloat16)
    r497_value = r497_value.to(tl.float32)
    r497_value = tl.broadcast_to(r497_value, [$(73728)])
    tl.store($(candidate_vto_copy_78_ptr) + r497_z, r497_value)
    r498_z = tl.arange($(73728))
    r498_value = (tl.load($(candidate_vto_copy_78_ptr) + ((((r498_z / $(256)) % $(288)) * $(256)) + (r498_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r498_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r498_value = r498_value.to(tl.float32)
    r498_value = tl.broadcast_to(r498_value, [$(73728)])
    tl.store($(candidate_vmul_54_ptr) + r498_z, r498_value)
    r499_z = tl.arange($(73728))
    r499_value = (tl.load($(candidate_vmul_54_ptr) + ((((r499_z / $(256)) % $(288)) * $(256)) + (r499_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_18_ptr) + ((((r499_z / $(256)) % $(288)) * $(256)) + (r499_z % $(256))), dtype=tl.float32))
    r499_value = r499_value.to(tl.float32)
    r499_value = tl.broadcast_to(r499_value, [$(73728)])
    tl.store($(candidate_vmul_55_ptr) + r499_z, r499_value)
    r500_z = tl.arange($(73728))
    r500_value = (tl.load($(candidate_vmul_54_ptr) + ((((r500_z / $(256)) % $(288)) * $(256)) + (r500_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r500_z / $(256)) % $(288)), dtype=tl.float32))
    r500_value = r500_value.to(tl.float32)
    r500_value = tl.broadcast_to(r500_value, [$(73728)])
    tl.store($(candidate_vmul_56_ptr) + r500_z, r500_value)
    r501_z = tl.arange($(288))
    r501_lane = tl.arange($(256))
    r501_value = tl.sum(tl.load($(candidate_vmul_55_ptr) + ((r501_z[:, None] * $(256)) + r501_lane[None, :]), dtype=tl.float32), axis=1)
    r501_value = r501_value.to(tl.float32)
    r501_value = tl.broadcast_to(r501_value, [$(288)])
    tl.store($(candidate_vsum_11_ptr) + r501_z, r501_value)
    r502_z = tl.arange($(288))
    r502_value = ((tl.load($(candidate_vrsqrt_ptr) + r502_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r502_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r502_z, dtype=tl.float32)))
    r502_value = r502_value.to(tl.float32)
    r502_value = tl.broadcast_to(r502_value, [$(288)])
    tl.store($(candidate_vpow_6_ptr) + r502_z, r502_value)
    r503_z = tl.arange($(288))
    r503_value = (tl.load($(candidate_vsum_11_ptr) + r503_z, dtype=tl.float32) * -0.5)
    r503_value = r503_value.to(tl.float32)
    r503_value = tl.broadcast_to(r503_value, [$(288)])
    tl.store($(candidate_vmul_57_ptr) + r503_z, r503_value)
    r504_z = tl.arange($(288))
    r504_value = (tl.load($(candidate_vmul_57_ptr) + r504_z, dtype=tl.float32) * tl.load($(candidate_vpow_6_ptr) + r504_z, dtype=tl.float32))
    r504_value = r504_value.to(tl.float32)
    r504_value = tl.broadcast_to(r504_value, [$(288)])
    tl.store($(candidate_vmul_58_ptr) + r504_z, r504_value)
    r505_z = tl.arange($(73728))
    r505_value = (tl.load($(candidate_vmul_58_ptr) + ((r505_z / $(256)) % $(288)), dtype=tl.float32) / 256.0)
    r505_value = r505_value.to(tl.float32)
    r505_value = tl.broadcast_to(r505_value, [$(73728)])
    tl.store($(candidate_vdiv_2_ptr) + r505_z, r505_value)
    r506_z = tl.arange($(73728))
    r506_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r506_z / $(256)) % $(288)) * $(256)) + (r506_z % $(256))), dtype=tl.float32)))
    r506_value = r506_value.to(tl.float32)
    r506_value = tl.broadcast_to(r506_value, [$(73728)])
    tl.store($(candidate_vpow_7_ptr) + r506_z, r506_value)
    r507_z = tl.arange($(73728))
    r507_value = (tl.load($(candidate_vpow_7_ptr) + ((((r507_z / $(256)) % $(288)) * $(256)) + (r507_z % $(256))), dtype=tl.float32) * 2.0)
    r507_value = r507_value.to(tl.float32)
    r507_value = tl.broadcast_to(r507_value, [$(73728)])
    tl.store($(candidate_vmul_59_ptr) + r507_z, r507_value)
    r508_z = tl.arange($(73728))
    r508_value = (tl.load($(candidate_vdiv_2_ptr) + ((((r508_z / $(256)) % $(288)) * $(256)) + (r508_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_59_ptr) + ((((r508_z / $(256)) % $(288)) * $(256)) + (r508_z % $(256))), dtype=tl.float32))
    r508_value = r508_value.to(tl.float32)
    r508_value = tl.broadcast_to(r508_value, [$(73728)])
    tl.store($(candidate_vmul_60_ptr) + r508_z, r508_value)
    r509_z = tl.arange($(73728))
    r509_value = (tl.load($(candidate_vmul_56_ptr) + ((((r509_z / $(256)) % $(288)) * $(256)) + (r509_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_60_ptr) + ((((r509_z / $(256)) % $(288)) * $(256)) + (r509_z % $(256))), dtype=tl.float32))
    r509_value = r509_value.to(tl.float32)
    r509_value = tl.broadcast_to(r509_value, [$(73728)])
    tl.store($(candidate_vadd_63_ptr) + r509_z, r509_value)
    r510_z = tl.arange($(73728))
    r510_value = tl.load($(candidate_vadd_63_ptr) + ((((r510_z / $(256)) % $(288)) * $(256)) + (r510_z % $(256))), dtype=tl.float32)
    r510_value = r510_value.to(tl.bfloat16)
    r510_value = tl.broadcast_to(r510_value, [$(73728)])
    tl.store($(candidate_vto_copy_79_ptr) + r510_z, r510_value)
    r511_row = tl.arange($(288))
    r511_col = tl.arange($(512))
    r511_lane = tl.arange($(256))
    r511_left = tl.load($(candidate_vto_copy_79_ptr) + ((((((r511_row[:, None] * $(256)) + r511_lane[None, :]) / $(256)) % $(288)) * $(256)) + (((r511_row[:, None] * $(256)) + r511_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r511_right = tl.load($(candidate_vto_copy_ptr) + (((r511_lane[:, None] % $(256)) * $(512)) + (r511_col[None, :] % $(512))), dtype=tl.bfloat16)
    r511_value = tl.dot(r511_left, r511_right)
    r511_value = r511_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_7_ptr) + ((r511_row[:, None] * $(512)) + r511_col[None, :]), r511_value)
    r512_z = tl.arange($(147456))
    r512_value = (tl.load($(candidate_vmm_5_ptr) + ((((r512_z / $(512)) % $(288)) * $(512)) + (r512_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_7_ptr) + ((((r512_z / $(512)) % $(288)) * $(512)) + (r512_z % $(512))), dtype=tl.bfloat16))
    r512_value = r512_value.to(tl.bfloat16)
    r512_value = tl.broadcast_to(r512_value, [$(147456)])
    tl.store($(candidate_vadd_64_ptr) + r512_z, r512_value)
    z = tl.arange($(147456))
    value = tl.load($(candidate_vadd_64_ptr) + ((((z / $(512)) % $(288)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(147456)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_seqused_dx
