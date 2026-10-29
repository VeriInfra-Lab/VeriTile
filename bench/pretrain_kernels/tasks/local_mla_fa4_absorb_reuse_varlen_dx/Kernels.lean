import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_reuse_varlen_dx
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
  let vsub_4_ptr : RegionName := "vsub_4"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_5_ptr : RegionName := "vsub_5"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
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
  let vcat_8_ptr : RegionName := "vcat_8"
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
  let vcat_10_ptr : RegionName := "vcat_10"
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
    r16_z = tl.arange($(128))
    r16_value = tl.where(r16_z < $(48), tl.cast((r16_z - $(0)), tl.int64), tl.where(r16_z < $(64), tl.cast((r16_z - $(48)), tl.int64), tl.cast((r16_z - $(64)), tl.int64)))
    r16_value = tl.broadcast_to(r16_value, [$(128)])
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
    r21_z = tl.arange($(4096))
    r21_value = (tl.cast(tl.toReal(((r21_z / $(32)) % $(128))), tl.float32) * tl.load($(vmul_7_ptr) + (r21_z % $(32)), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(4096)])
    tl.store($(vmul_8_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(8192))
    r22_value = tl.where((r22_z % $(64)) < $(32), tl.load($(vmul_8_ptr) + ((((r22_z / $(64)) % $(128)) * $(32)) + (((r22_z % $(64)) - $(0)) % $(32))), dtype=tl.float32), tl.load($(vmul_8_ptr) + ((((r22_z / $(64)) % $(128)) * $(32)) + (((r22_z % $(64)) - $(32)) % $(32))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(8192)])
    tl.store($(vcat_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(8192))
    r23_value = tl.cos(tl.load($(vcat_1_ptr) + ((((r23_z / $(64)) % $(128)) * $(64)) + (r23_z % $(64))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(8192)])
    tl.store($(vcos_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(8192))
    r24_value = tl.load($(vcos_ptr) + (((((((tl.load($(vcat_ptr) + ((r24_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r24_z % $(64))) / $(64)) % $(128)) % $(128)) * $(64)) + ((((tl.load($(vcat_ptr) + ((r24_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r24_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(8192)])
    tl.store($(vindex_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(8192))
    r25_value = tl.sin(tl.load($(vcat_1_ptr) + ((((r25_z / $(64)) % $(128)) * $(64)) + (r25_z % $(64))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(8192)])
    tl.store($(vsin_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(8192))
    r26_value = tl.load($(vsin_ptr) + (((((((tl.load($(vcat_ptr) + ((r26_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r26_z % $(64))) / $(64)) % $(128)) % $(128)) * $(64)) + ((((tl.load($(vcat_ptr) + ((r26_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r26_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(8192)])
    tl.store($(vindex_1_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(8192))
    r27_value = tl.load($(vindex_ptr) + ((((r27_z / $(64)) % $(128)) * $(64)) + (r27_z % $(64))), dtype=tl.float32)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(8192)])
    tl.store($(vto_copy_14_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(8192))
    r28_value = tl.load($(vindex_1_ptr) + ((((r28_z / $(64)) % $(128)) * $(64)) + (r28_z % $(64))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(8192)])
    tl.store($(vto_copy_15_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(1))
    r29_value = (tl.load($(vcat_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r29_value = tl.broadcast_to(r29_value, [$(1)])
    tl.store($(vsub_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(129))
    r30_value = tl.where(r30_z < $(1), tl.load($(vsub_ptr) + $(0), dtype=tl.int64), tl.load($(vcat_ptr) + ((r30_z - $(1)) % $(128)), dtype=tl.int64))
    r30_value = tl.broadcast_to(r30_value, [$(129)])
    tl.store($(vcat_2_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(128))
    r31_value = (tl.load($(vcat_2_ptr) + (r31_z + $(1)), dtype=tl.int64) - tl.load($(vcat_2_ptr) + r31_z, dtype=tl.int64))
    r31_value = tl.broadcast_to(r31_value, [$(128)])
    tl.store($(vsub_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(128))
    r32_value = (tl.load($(vsub_1_ptr) + r32_z, dtype=tl.int64) != tl.full([], $(1), dtype=tl.int64))
    r32_value = tl.broadcast_to(r32_value, [$(128)])
    tl.store($(vne_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(128))
    r33_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(128)])
    for r33_lane in range($(128)) {
      r33_value = tl.where(r33_lane <= r33_z, (r33_value + tl.where(tl.load($(vne_ptr) + (r33_lane % $(128)), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))), r33_value)
    }
    r33_value = tl.broadcast_to(r33_value, [$(128)])
    tl.store($(vcumsum_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(16384))
    r34_value = (tl.load($(vcumsum_1_ptr) + ((r34_z / $(128)) % $(128)), dtype=tl.int64) == tl.load($(vcumsum_1_ptr) + (r34_z % $(128)), dtype=tl.int64))
    r34_value = tl.broadcast_to(r34_value, [$(16384)])
    tl.store($(veq_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(16384))
    r35_value = tl.where((r35_z % $(128)) <= ((r35_z / $(128)) % $(128)) + $(0), true, false)
    r35_value = tl.broadcast_to(r35_value, [$(16384)])
    tl.store($(vtril_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(16384))
    r36_value = (tl.load($(veq_ptr) + ((((r36_z / $(128)) % $(128)) * $(128)) + (r36_z % $(128))), dtype=tl.int1) & tl.load($(vtril_ptr) + ((((r36_z / $(128)) % $(128)) * $(128)) + (r36_z % $(128))), dtype=tl.int1))
    r36_value = tl.broadcast_to(r36_value, [$(16384)])
    tl.store($(vbitwise_and_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(16384))
    r37_value = tl.where(tl.load($(vbitwise_and_ptr) + ((((r37_z / $(128)) % $(128)) * $(128)) + (r37_z % $(128))), dtype=tl.int1), tl.cast(0.0, tl.bfloat16), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16))
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(16384)])
    tl.store($(vwhere_ptr) + r37_z, r37_value)
    r38_row = tl.arange($(128))
    r38_col = tl.arange($(256))
    r38_lane = tl.arange($(512))
    r38_left = tl.load($(input0_ptr) + ((((((r38_row[:, None] * $(512)) + r38_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r38_row[:, None] * $(512)) + r38_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r38_right = tl.load($(vto_copy_ptr) + (((r38_col[None, :] % $(256)) * $(512)) + (r38_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r38_value = tl.dot(r38_left, r38_right)
    r38_value = r38_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r38_row[:, None] * $(256)) + r38_col[None, :]), r38_value)
    r39_z = tl.arange($(32768))
    r39_value = tl.load($(vmm_ptr) + ((((r39_z / $(256)) % $(128)) * $(256)) + (r39_z % $(256))), dtype=tl.bfloat16)
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(32768)])
    tl.store($(vto_copy_18_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(32768))
    r40_value = ((tl.load($(vto_copy_18_ptr) + ((((r40_z / $(256)) % $(128)) * $(256)) + (r40_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_18_ptr) + ((((r40_z / $(256)) % $(128)) * $(256)) + (r40_z % $(256))), dtype=tl.float32)))
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(32768)])
    tl.store($(vpow_2_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(128))
    r41_lane = tl.arange($(256))
    r41_value = (tl.sum(tl.load($(vpow_2_ptr) + ((r41_z[:, None] * $(256)) + r41_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(128)])
    tl.store($(vmean_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(128))
    r42_value = (tl.load($(vmean_ptr) + r42_z, dtype=tl.float32) + 0.000001)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(128)])
    tl.store($(vadd_2_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(128))
    r43_value = tl.rsqrt(tl.load($(vadd_2_ptr) + r43_z, dtype=tl.float32))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(128)])
    tl.store($(vrsqrt_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(32768))
    r44_value = (tl.load($(vto_copy_18_ptr) + ((((r44_z / $(256)) % $(128)) * $(256)) + (r44_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((r44_z / $(256)) % $(128)), dtype=tl.float32))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(32768)])
    tl.store($(vmul_9_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(32768))
    r45_value = (tl.cast(tl.load($(vto_copy_1_ptr) + (r45_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_9_ptr) + ((((r45_z / $(256)) % $(128)) * $(256)) + (r45_z % $(256))), dtype=tl.float32))
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(32768)])
    tl.store($(vmul_10_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(32768))
    r46_value = tl.load($(vmul_10_ptr) + ((((r46_z / $(256)) % $(128)) * $(256)) + (r46_z % $(256))), dtype=tl.float32)
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(32768)])
    tl.store($(vto_copy_19_ptr) + r46_z, r46_value)
    r47_row = tl.arange($(128))
    r47_col = tl.arange($(1536))
    r47_lane = tl.arange($(256))
    r47_left = tl.load($(vto_copy_19_ptr) + ((((((r47_row[:, None] * $(256)) + r47_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r47_row[:, None] * $(256)) + r47_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r47_right = tl.load($(vto_copy_2_ptr) + (((r47_col[None, :] % $(1536)) * $(256)) + (r47_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r47_value = tl.dot(r47_left, r47_right)
    r47_value = r47_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r47_row[:, None] * $(1536)) + r47_col[None, :]), r47_value)
    r48_row = tl.arange($(128))
    r48_col = tl.arange($(576))
    r48_lane = tl.arange($(512))
    r48_left = tl.load($(input0_ptr) + ((((((r48_row[:, None] * $(512)) + r48_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r48_row[:, None] * $(512)) + r48_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r48_right = tl.load($(vto_copy_3_ptr) + (((r48_col[None, :] % $(576)) * $(512)) + (r48_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r48_value = tl.dot(r48_left, r48_right)
    r48_value = r48_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r48_row[:, None] * $(576)) + r48_col[None, :]), r48_value)
    r49_z = tl.arange($(65536))
    r49_value = tl.load($(vmm_2_ptr) + ((((r49_z / $(512)) % $(128)) * $(576)) + (r49_z % $(512))), dtype=tl.bfloat16)
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(65536)])
    tl.store($(vto_copy_20_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(65536))
    r50_value = ((tl.load($(vto_copy_20_ptr) + ((((r50_z / $(512)) % $(128)) * $(512)) + (r50_z % $(512))), dtype=tl.float32)) * (tl.load($(vto_copy_20_ptr) + ((((r50_z / $(512)) % $(128)) * $(512)) + (r50_z % $(512))), dtype=tl.float32)))
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(65536)])
    tl.store($(vpow_3_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(128))
    r51_lane = tl.arange($(512))
    r51_value = (tl.sum(tl.load($(vpow_3_ptr) + ((r51_z[:, None] * $(512)) + r51_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(128)])
    tl.store($(vmean_1_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(128))
    r52_value = (tl.load($(vmean_1_ptr) + r52_z, dtype=tl.float32) + 0.000001)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(128)])
    tl.store($(vadd_3_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(128))
    r53_value = tl.rsqrt(tl.load($(vadd_3_ptr) + r53_z, dtype=tl.float32))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(128)])
    tl.store($(vrsqrt_1_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(65536))
    r54_value = (tl.load($(vto_copy_20_ptr) + ((((r54_z / $(512)) % $(128)) * $(512)) + (r54_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r54_z / $(512)) % $(128)), dtype=tl.float32))
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(65536)])
    tl.store($(vmul_11_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(65536))
    r55_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r55_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_11_ptr) + ((((r55_z / $(512)) % $(128)) * $(512)) + (r55_z % $(512))), dtype=tl.float32))
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(65536)])
    tl.store($(vmul_12_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(65536))
    r56_value = tl.load($(vmul_12_ptr) + ((((r56_z / $(512)) % $(128)) * $(512)) + (r56_z % $(512))), dtype=tl.float32)
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(65536)])
    tl.store($(vto_copy_21_ptr) + r56_z, r56_value)
    r57_row = tl.arange($(128))
    r57_col = tl.arange($(2048))
    r57_lane = tl.arange($(512))
    r57_left = tl.load($(vto_copy_21_ptr) + ((((((r57_row[:, None] * $(512)) + r57_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r57_row[:, None] * $(512)) + r57_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r57_right = tl.load($(vto_copy_5_ptr) + (((r57_col[None, :] % $(2048)) * $(512)) + (r57_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r57_value = tl.dot(r57_left, r57_right)
    r57_value = r57_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r57_row[:, None] * $(2048)) + r57_col[None, :]), r57_value)
    r58_z = tl.arange($(32768))
    r58_value = (tl.load($(vmm_1_ptr) + ((((r58_z / $(32)) % $(128)) * $(1536)) + (((r58_z / $(4096)) % $(8)) * $(192)) + ((r58_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r58_z / $(32)) % $(128)) * $(64)) + (r58_z % $(32))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(32768)])
    tl.store($(vmul_13_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(32768))
    r59_value = (tl.load($(vmm_1_ptr) + ((((r59_z / $(32)) % $(128)) * $(1536)) + (((r59_z / $(4096)) % $(8)) * $(192)) + ((r59_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r59_z / $(32)) % $(128)) * $(64)) + (r59_z % $(32))), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(32768)])
    tl.store($(vmul_14_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(32768))
    r60_value = (tl.load($(vmul_13_ptr) + ((((r60_z / $(4096)) % $(8)) * $(4096)) + (((r60_z / $(32)) % $(128)) * $(32)) + (r60_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_14_ptr) + ((((r60_z / $(4096)) % $(8)) * $(4096)) + (((r60_z / $(32)) % $(128)) * $(32)) + (r60_z % $(32))), dtype=tl.bfloat16))
    r60_value = r60_value.to(tl.bfloat16)
    r60_value = tl.broadcast_to(r60_value, [$(32768)])
    tl.store($(vsub_4_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(32768))
    r61_value = (tl.load($(vmm_1_ptr) + ((((r61_z / $(32)) % $(128)) * $(1536)) + (((r61_z / $(4096)) % $(8)) * $(192)) + ((r61_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r61_z / $(32)) % $(128)) * $(64)) + (r61_z % $(32))), dtype=tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(32768)])
    tl.store($(vmul_15_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(32768))
    r62_value = (tl.load($(vmm_1_ptr) + ((((r62_z / $(32)) % $(128)) * $(1536)) + (((r62_z / $(4096)) % $(8)) * $(192)) + ((r62_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r62_z / $(32)) % $(128)) * $(64)) + (r62_z % $(32))), dtype=tl.bfloat16))
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(32768)])
    tl.store($(vmul_16_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(32768))
    r63_value = (tl.load($(vmul_15_ptr) + ((((r63_z / $(4096)) % $(8)) * $(4096)) + (((r63_z / $(32)) % $(128)) * $(32)) + (r63_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_16_ptr) + ((((r63_z / $(4096)) % $(8)) * $(4096)) + (((r63_z / $(32)) % $(128)) * $(32)) + (r63_z % $(32))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(32768)])
    tl.store($(vadd_4_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(65536))
    r64_value = tl.where((r64_z % $(64)) < $(32), tl.load($(vsub_4_ptr) + ((((r64_z / $(8192)) % $(8)) * $(4096)) + (((r64_z / $(64)) % $(128)) * $(32)) + (((r64_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_4_ptr) + ((((r64_z / $(8192)) % $(8)) * $(4096)) + (((r64_z / $(64)) % $(128)) * $(32)) + (((r64_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(65536)])
    tl.store($(vcat_4_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(4096))
    r65_value = (tl.load($(vmm_2_ptr) + ((((r65_z / $(32)) % $(128)) * $(576)) + ((r65_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r65_z / $(32)) % $(128)) * $(64)) + (r65_z % $(32))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(4096)])
    tl.store($(vmul_17_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(4096))
    r66_value = (tl.load($(vmm_2_ptr) + ((((r66_z / $(32)) % $(128)) * $(576)) + ((r66_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r66_z / $(32)) % $(128)) * $(64)) + (r66_z % $(32))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(4096)])
    tl.store($(vmul_18_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(4096))
    r67_value = (tl.load($(vmul_17_ptr) + ((((r67_z / $(32)) % $(128)) * $(32)) + (r67_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_18_ptr) + ((((r67_z / $(32)) % $(128)) * $(32)) + (r67_z % $(32))), dtype=tl.bfloat16))
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(4096)])
    tl.store($(vsub_5_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(4096))
    r68_value = (tl.load($(vmm_2_ptr) + ((((r68_z / $(32)) % $(128)) * $(576)) + ((r68_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r68_z / $(32)) % $(128)) * $(64)) + (r68_z % $(32))), dtype=tl.bfloat16))
    r68_value = r68_value.to(tl.bfloat16)
    r68_value = tl.broadcast_to(r68_value, [$(4096)])
    tl.store($(vmul_19_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(4096))
    r69_value = (tl.load($(vmm_2_ptr) + ((((r69_z / $(32)) % $(128)) * $(576)) + ((r69_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r69_z / $(32)) % $(128)) * $(64)) + (r69_z % $(32))), dtype=tl.bfloat16))
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(4096)])
    tl.store($(vmul_20_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(4096))
    r70_value = (tl.load($(vmul_19_ptr) + ((((r70_z / $(32)) % $(128)) * $(32)) + (r70_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_20_ptr) + ((((r70_z / $(32)) % $(128)) * $(32)) + (r70_z % $(32))), dtype=tl.bfloat16))
    r70_value = r70_value.to(tl.bfloat16)
    r70_value = tl.broadcast_to(r70_value, [$(4096)])
    tl.store($(vadd_5_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(8192))
    r71_value = tl.where((r71_z % $(64)) < $(32), tl.load($(vsub_5_ptr) + ((((r71_z / $(64)) % $(128)) * $(32)) + (((r71_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r71_z / $(64)) % $(128)) * $(32)) + (((r71_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r71_value = r71_value.to(tl.bfloat16)
    r71_value = tl.broadcast_to(r71_value, [$(8192)])
    tl.store($(vcat_5_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(196608))
    r72_value = tl.where((r72_z % $(192)) < $(128), tl.load($(vmm_1_ptr) + ((((((((r72_z / $(192)) % $(128)) * $(1536)) + (((r72_z / $(24576)) % $(8)) * $(192)) + ((r72_z % $(192)) - $(0))) / $(1536)) % $(128)) * $(1536)) + (((((r72_z / $(192)) % $(128)) * $(1536)) + (((r72_z / $(24576)) % $(8)) * $(192)) + ((r72_z % $(192)) - $(0))) % $(1536))), dtype=tl.bfloat16), tl.load($(vcat_4_ptr) + ((((r72_z / $(24576)) % $(8)) * $(8192)) + (((r72_z / $(192)) % $(128)) * $(64)) + (((r72_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r72_value = r72_value.to(tl.bfloat16)
    r72_value = tl.broadcast_to(r72_value, [$(196608)])
    tl.store($(vcat_6_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(196608))
    r73_value = tl.where((r73_z % $(192)) < $(128), tl.load($(vmm_3_ptr) + ((((((((r73_z / $(192)) % $(128)) * $(2048)) + (((r73_z / $(24576)) % $(8)) * $(256)) + ((r73_z % $(192)) - $(0))) / $(2048)) % $(128)) * $(2048)) + (((((r73_z / $(192)) % $(128)) * $(2048)) + (((r73_z / $(24576)) % $(8)) * $(256)) + ((r73_z % $(192)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_5_ptr) + ((((r73_z / $(192)) % $(128)) * $(64)) + (((r73_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(196608)])
    tl.store($(vcat_7_ptr) + r73_z, r73_value)
    r74_row = tl.arange($(128))
    r74_col = tl.arange($(128))
    r74_lane = tl.arange($(192))
    for r74_group in range($(8)) {
      r74_left = tl.load($(vcat_6_ptr) + (((((((r74_group % $(8)) * $(24576)) + (r74_row[:, None] * $(192)) + r74_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r74_group % $(8)) * $(24576)) + (r74_row[:, None] * $(192)) + r74_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r74_group % $(8)) * $(24576)) + (r74_row[:, None] * $(192)) + r74_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r74_right = tl.load($(vcat_7_ptr) + (((((((r74_group % $(8)) * $(24576)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r74_group % $(8)) * $(24576)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) % $(128)) * $(192)) + (((((r74_group % $(8)) * $(24576)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r74_value = tl.dot(r74_left, r74_right)
      r74_value = r74_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r74_group * $(16384)) + (r74_row[:, None] * $(128)) + r74_col[None, :]), r74_value)
    }
    r75_z = tl.arange($(131072))
    r75_value = (tl.load($(vbmm_ptr) + ((((r75_z / $(16384)) % $(8)) * $(16384)) + (((r75_z / $(128)) % $(128)) * $(128)) + (r75_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r75_value = r75_value.to(tl.bfloat16)
    r75_value = tl.broadcast_to(r75_value, [$(131072)])
    tl.store($(vmul_21_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(131072))
    r76_value = (tl.load($(vmul_21_ptr) + ((((r76_z / $(16384)) % $(8)) * $(16384)) + (((r76_z / $(128)) % $(128)) * $(128)) + (r76_z % $(128))), dtype=tl.bfloat16) + tl.load($(vwhere_ptr) + ((((r76_z / $(128)) % $(128)) * $(128)) + (r76_z % $(128))), dtype=tl.bfloat16))
    r76_value = r76_value.to(tl.bfloat16)
    r76_value = tl.broadcast_to(r76_value, [$(131072)])
    tl.store($(vadd_6_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(131072))
    r77_value = tl.load($(vadd_6_ptr) + ((((r77_z / $(16384)) % $(8)) * $(16384)) + (((r77_z / $(128)) % $(128)) * $(128)) + (r77_z % $(128))), dtype=tl.bfloat16)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(131072)])
    tl.store($(vto_copy_22_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(131072))
    r78_lane = tl.arange($(128))
    r78_row_values = tl.load($(vto_copy_22_ptr) + ((((r78_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r78_z[:, None] / $(128)) % $(128)) * $(128)) + (r78_lane[None, :] % $(128))), dtype=tl.float32)
    r78_maximum = tl.max(r78_row_values, axis=1, keep_dims=true)
    r78_weights = tl.exp(r78_row_values - r78_maximum)
    r78_total = tl.sum(r78_weights, axis=1)
    r78_maximum = tl.max(r78_row_values, axis=1)
    r78_value = tl.exp(tl.load($(vto_copy_22_ptr) + ((((r78_z / $(16384)) % $(8)) * $(16384)) + (((r78_z / $(128)) % $(128)) * $(128)) + (r78_z % $(128))), dtype=tl.float32) - r78_maximum) / r78_total
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(131072)])
    tl.store($(vsoftmax_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(131072))
    r79_value = tl.load($(vsoftmax_ptr) + ((((r79_z / $(16384)) % $(8)) * $(16384)) + (((r79_z / $(128)) % $(128)) * $(128)) + (r79_z % $(128))), dtype=tl.float32)
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(131072)])
    tl.store($(vto_copy_23_ptr) + r79_z, r79_value)
    r80_row = tl.arange($(128))
    r80_col = tl.arange($(1024))
    r80_lane = tl.arange($(512))
    r80_left = tl.load($(input1_ptr) + ((((((r80_row[:, None] * $(512)) + r80_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r80_row[:, None] * $(512)) + r80_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r80_right = tl.load($(vto_copy_6_ptr) + (((r80_lane[:, None] % $(512)) * $(1024)) + (r80_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r80_value = tl.dot(r80_left, r80_right)
    r80_value = r80_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r80_row[:, None] * $(1024)) + r80_col[None, :]), r80_value)
    r81_row = tl.arange($(128))
    r81_col = tl.arange($(128))
    r81_lane = tl.arange($(128))
    for r81_group in range($(8)) {
      r81_left = tl.load($(vto_copy_23_ptr) + (((((((r81_group % $(8)) * $(16384)) + (r81_lane[None, :] * $(128)) + r81_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r81_group % $(8)) * $(16384)) + (r81_lane[None, :] * $(128)) + r81_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r81_group % $(8)) * $(16384)) + (r81_lane[None, :] * $(128)) + r81_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r81_right = tl.load($(vmm_5_ptr) + (((((((r81_group % $(8)) * $(16384)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r81_group % $(8)) * $(16384)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r81_group % $(8)) * $(16384)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r81_value = tl.dot(r81_left, r81_right)
      r81_value = r81_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r81_group * $(16384)) + (r81_row[:, None] * $(128)) + r81_col[None, :]), r81_value)
    }
    r82_row = tl.arange($(128))
    r82_col = tl.arange($(128))
    r82_lane = tl.arange($(128))
    for r82_group in range($(8)) {
      r82_left = tl.load($(vmm_5_ptr) + (((((((r82_group % $(8)) * $(16384)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r82_group % $(8)) * $(16384)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r82_group % $(8)) * $(16384)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r82_right = tl.load($(vmm_3_ptr) + (((((((r82_group % $(8)) * $(16384)) + (r82_col[None, :] * $(128)) + r82_lane[:, None]) / $(128)) % $(128)) * $(2048)) + ((((((r82_group % $(8)) * $(16384)) + (r82_col[None, :] * $(128)) + r82_lane[:, None]) / $(16384)) % $(8)) * $(256)) + ((((r82_group % $(8)) * $(16384)) + (r82_col[None, :] * $(128)) + r82_lane[:, None]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r82_value = tl.dot(r82_left, r82_right)
      r82_value = r82_value.to(tl.bfloat16)
      tl.store($(vbmm_3_ptr) + ((r82_group * $(16384)) + (r82_row[:, None] * $(128)) + r82_col[None, :]), r82_value)
    }
    r83_z = tl.arange($(131072))
    r83_value = tl.load($(vbmm_3_ptr) + ((((r83_z / $(16384)) % $(8)) * $(16384)) + (((r83_z / $(128)) % $(128)) * $(128)) + (r83_z % $(128))), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(131072)])
    tl.store($(vto_copy_24_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(131072))
    r84_lane = tl.arange($(128))
    r84_row_values = tl.load($(vsoftmax_ptr) + ((((r84_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r84_z[:, None] / $(128)) % $(128)) * $(128)) + (r84_lane[None, :] % $(128))), dtype=tl.float32)
    r84_row_grad = tl.load($(vto_copy_24_ptr) + ((((r84_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r84_z[:, None] / $(128)) % $(128)) * $(128)) + (r84_lane[None, :] % $(128))), dtype=tl.float32)
    r84_weighted = r84_row_grad * r84_row_values
    r84_dot_grad = tl.sum(r84_weighted, axis=1)
    r84_value = (tl.load($(vto_copy_24_ptr) + ((((r84_z / $(16384)) % $(8)) * $(16384)) + (((r84_z / $(128)) % $(128)) * $(128)) + (r84_z % $(128))), dtype=tl.float32) - r84_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r84_z / $(16384)) % $(8)) * $(16384)) + (((r84_z / $(128)) % $(128)) * $(128)) + (r84_z % $(128))), dtype=tl.float32)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(131072)])
    tl.store($(vsoftmax_backward_data_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(131072))
    r85_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r85_z / $(16384)) % $(8)) * $(16384)) + (((r85_z / $(128)) % $(128)) * $(128)) + (r85_z % $(128))), dtype=tl.float32)
    r85_value = r85_value.to(tl.bfloat16)
    r85_value = tl.broadcast_to(r85_value, [$(131072)])
    tl.store($(vto_copy_25_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(131072))
    r86_value = (tl.load($(vto_copy_25_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(128)) % $(128)) * $(128)) + (r86_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r86_value = r86_value.to(tl.bfloat16)
    r86_value = tl.broadcast_to(r86_value, [$(131072)])
    tl.store($(vmul_22_ptr) + r86_z, r86_value)
    r87_row = tl.arange($(192))
    r87_col = tl.arange($(128))
    r87_lane = tl.arange($(128))
    for r87_group in range($(8)) {
      r87_left = tl.load($(vcat_6_ptr) + (((((((r87_group % $(8)) * $(24576)) + (r87_lane[None, :] * $(192)) + r87_row[:, None]) / $(24576)) % $(8)) * $(24576)) + ((((((r87_group % $(8)) * $(24576)) + (r87_lane[None, :] * $(192)) + r87_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r87_group % $(8)) * $(24576)) + (r87_lane[None, :] * $(192)) + r87_row[:, None]) % $(192))), dtype=tl.bfloat16)
      r87_right = tl.load($(vmul_22_ptr) + (((((((r87_group % $(8)) * $(16384)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r87_group % $(8)) * $(16384)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r87_group % $(8)) * $(16384)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r87_value = tl.dot(r87_left, r87_right)
      r87_value = r87_value.to(tl.bfloat16)
      tl.store($(vbmm_4_ptr) + ((r87_group * $(24576)) + (r87_row[:, None] * $(128)) + r87_col[None, :]), r87_value)
    }
    r88_row = tl.arange($(128))
    r88_col = tl.arange($(192))
    r88_lane = tl.arange($(128))
    for r88_group in range($(8)) {
      r88_left = tl.load($(vmul_22_ptr) + (((((((r88_group % $(8)) * $(16384)) + (r88_row[:, None] * $(128)) + r88_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r88_group % $(8)) * $(16384)) + (r88_row[:, None] * $(128)) + r88_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r88_group % $(8)) * $(16384)) + (r88_row[:, None] * $(128)) + r88_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r88_right = tl.load($(vcat_7_ptr) + (((((((r88_group % $(8)) * $(24576)) + (r88_col[None, :] * $(128)) + r88_lane[:, None]) / $(24576)) % $(8)) * $(24576)) + (((((r88_group % $(8)) * $(24576)) + (r88_col[None, :] * $(128)) + r88_lane[:, None]) % $(128)) * $(192)) + (((((r88_group % $(8)) * $(24576)) + (r88_col[None, :] * $(128)) + r88_lane[:, None]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r88_value = tl.dot(r88_left, r88_right)
      r88_value = r88_value.to(tl.bfloat16)
      tl.store($(vbmm_5_ptr) + ((r88_group * $(24576)) + (r88_row[:, None] * $(192)) + r88_col[None, :]), r88_value)
    }
    r89_z = tl.arange($(8192))
    r89_lane = tl.arange($(8))
    r89_value = tl.sum(tl.load($(vbmm_4_ptr) + ((r89_lane[None, :] * $(24576)) + ((r89_z[:, None] % $(64)) * $(128)) + ((r89_z[:, None] / $(64)) % $(128)) + $(16384)), dtype=tl.bfloat16), axis=1)
    r89_value = r89_value.to(tl.bfloat16)
    r89_value = tl.broadcast_to(r89_value, [$(8192)])
    tl.store($(vsum_1_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(4096))
    r90_value = (tl.load($(vsum_1_ptr) + ((((r90_z / $(32)) % $(128)) * $(64)) + (r90_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r90_z / $(32)) % $(128)) * $(64)) + (r90_z % $(32))), dtype=tl.bfloat16))
    r90_value = r90_value.to(tl.bfloat16)
    r90_value = tl.broadcast_to(r90_value, [$(4096)])
    tl.store($(vmul_23_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(4096))
    r91_value = (tl.load($(vsum_1_ptr) + ((((r91_z / $(32)) % $(128)) * $(64)) + (r91_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r91_z / $(32)) % $(128)) * $(64)) + (r91_z % $(32))), dtype=tl.bfloat16))
    r91_value = r91_value.to(tl.bfloat16)
    r91_value = tl.broadcast_to(r91_value, [$(4096)])
    tl.store($(vmul_24_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(4096))
    r92_value = -tl.load($(vsum_1_ptr) + ((((r92_z / $(32)) % $(128)) * $(64)) + (r92_z % $(32))), dtype=tl.bfloat16)
    r92_value = r92_value.to(tl.bfloat16)
    r92_value = tl.broadcast_to(r92_value, [$(4096)])
    tl.store($(vneg_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(4096))
    r93_value = (tl.load($(vneg_ptr) + ((((r93_z / $(32)) % $(128)) * $(32)) + (r93_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r93_z / $(32)) % $(128)) * $(64)) + (r93_z % $(32))), dtype=tl.bfloat16))
    r93_value = r93_value.to(tl.bfloat16)
    r93_value = tl.broadcast_to(r93_value, [$(4096)])
    tl.store($(vmul_25_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(4096))
    r94_value = (tl.load($(vmul_24_ptr) + ((((r94_z / $(32)) % $(128)) * $(32)) + (r94_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_25_ptr) + ((((r94_z / $(32)) % $(128)) * $(32)) + (r94_z % $(32))), dtype=tl.bfloat16))
    r94_value = r94_value.to(tl.bfloat16)
    r94_value = tl.broadcast_to(r94_value, [$(4096)])
    tl.store($(vadd_7_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(4096))
    r95_value = (tl.load($(vsum_1_ptr) + ((((r95_z / $(32)) % $(128)) * $(64)) + (r95_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r95_z / $(32)) % $(128)) * $(64)) + (r95_z % $(32))), dtype=tl.bfloat16))
    r95_value = r95_value.to(tl.bfloat16)
    r95_value = tl.broadcast_to(r95_value, [$(4096)])
    tl.store($(vmul_26_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(4096))
    r96_value = (tl.load($(vmul_23_ptr) + ((((r96_z / $(32)) % $(128)) * $(32)) + (r96_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_26_ptr) + ((((r96_z / $(32)) % $(128)) * $(32)) + (r96_z % $(32))), dtype=tl.bfloat16))
    r96_value = r96_value.to(tl.bfloat16)
    r96_value = tl.broadcast_to(r96_value, [$(4096)])
    tl.store($(vadd_8_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(32768))
    r97_value = (tl.load($(vbmm_5_ptr) + ((((r97_z / $(4096)) % $(8)) * $(24576)) + (((r97_z / $(32)) % $(128)) * $(192)) + (r97_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r97_z / $(32)) % $(128)) * $(64)) + (r97_z % $(32))), dtype=tl.bfloat16))
    r97_value = r97_value.to(tl.bfloat16)
    r97_value = tl.broadcast_to(r97_value, [$(32768)])
    tl.store($(vmul_27_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(32768))
    r98_value = (tl.load($(vbmm_5_ptr) + ((((r98_z / $(4096)) % $(8)) * $(24576)) + (((r98_z / $(32)) % $(128)) * $(192)) + (r98_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r98_z / $(32)) % $(128)) * $(64)) + (r98_z % $(32))), dtype=tl.bfloat16))
    r98_value = r98_value.to(tl.bfloat16)
    r98_value = tl.broadcast_to(r98_value, [$(32768)])
    tl.store($(vmul_28_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(32768))
    r99_value = -tl.load($(vbmm_5_ptr) + ((((r99_z / $(4096)) % $(8)) * $(24576)) + (((r99_z / $(32)) % $(128)) * $(192)) + (r99_z % $(32)) + $(128)), dtype=tl.bfloat16)
    r99_value = r99_value.to(tl.bfloat16)
    r99_value = tl.broadcast_to(r99_value, [$(32768)])
    tl.store($(vneg_1_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(32768))
    r100_value = (tl.load($(vneg_1_ptr) + ((((r100_z / $(4096)) % $(8)) * $(4096)) + (((r100_z / $(32)) % $(128)) * $(32)) + (r100_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r100_z / $(32)) % $(128)) * $(64)) + (r100_z % $(32))), dtype=tl.bfloat16))
    r100_value = r100_value.to(tl.bfloat16)
    r100_value = tl.broadcast_to(r100_value, [$(32768)])
    tl.store($(vmul_29_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(32768))
    r101_value = (tl.load($(vmul_28_ptr) + ((((r101_z / $(4096)) % $(8)) * $(4096)) + (((r101_z / $(32)) % $(128)) * $(32)) + (r101_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_29_ptr) + ((((r101_z / $(4096)) % $(8)) * $(4096)) + (((r101_z / $(32)) % $(128)) * $(32)) + (r101_z % $(32))), dtype=tl.bfloat16))
    r101_value = r101_value.to(tl.bfloat16)
    r101_value = tl.broadcast_to(r101_value, [$(32768)])
    tl.store($(vadd_9_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(32768))
    r102_value = (tl.load($(vbmm_5_ptr) + ((((r102_z / $(4096)) % $(8)) * $(24576)) + (((r102_z / $(32)) % $(128)) * $(192)) + (r102_z % $(32)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r102_z / $(32)) % $(128)) * $(64)) + (r102_z % $(32))), dtype=tl.bfloat16))
    r102_value = r102_value.to(tl.bfloat16)
    r102_value = tl.broadcast_to(r102_value, [$(32768)])
    tl.store($(vmul_30_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(32768))
    r103_value = (tl.load($(vmul_27_ptr) + ((((r103_z / $(4096)) % $(8)) * $(4096)) + (((r103_z / $(32)) % $(128)) * $(32)) + (r103_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_30_ptr) + ((((r103_z / $(4096)) % $(8)) * $(4096)) + (((r103_z / $(32)) % $(128)) * $(32)) + (r103_z % $(32))), dtype=tl.bfloat16))
    r103_value = r103_value.to(tl.bfloat16)
    r103_value = tl.broadcast_to(r103_value, [$(32768)])
    tl.store($(vadd_10_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(8192))
    r104_value = tl.where((((r104_z % $(64)) >= $(1)) & ((r104_z % $(64)) < $(64)) & (((r104_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_7_ptr) + ((((r104_z / $(64)) % $(128)) * $(32)) + ((((r104_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(8192)])
    tl.store($(vslice_backward_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(8192))
    r105_value = tl.where((((r105_z % $(64)) >= $(0)) & ((r105_z % $(64)) < $(64)) & (((r105_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_8_ptr) + ((((r105_z / $(64)) % $(128)) * $(32)) + ((((r105_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(8192)])
    tl.store($(vslice_backward_1_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(8192))
    r106_value = (tl.load($(vslice_backward_ptr) + ((((r106_z / $(64)) % $(128)) * $(64)) + (r106_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_1_ptr) + ((((r106_z / $(64)) % $(128)) * $(64)) + (r106_z % $(64))), dtype=tl.bfloat16))
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(8192)])
    tl.store($(vadd_11_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(65536))
    r107_value = tl.where((((r107_z % $(64)) >= $(1)) & ((r107_z % $(64)) < $(64)) & (((r107_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_9_ptr) + ((((r107_z / $(8192)) % $(8)) * $(4096)) + (((r107_z / $(64)) % $(128)) * $(32)) + ((((r107_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r107_value = r107_value.to(tl.bfloat16)
    r107_value = tl.broadcast_to(r107_value, [$(65536)])
    tl.store($(vslice_backward_2_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(65536))
    r108_value = tl.where((((r108_z % $(64)) >= $(0)) & ((r108_z % $(64)) < $(64)) & (((r108_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_10_ptr) + ((((r108_z / $(8192)) % $(8)) * $(4096)) + (((r108_z / $(64)) % $(128)) * $(32)) + ((((r108_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r108_value = r108_value.to(tl.bfloat16)
    r108_value = tl.broadcast_to(r108_value, [$(65536)])
    tl.store($(vslice_backward_3_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(65536))
    r109_value = (tl.load($(vslice_backward_2_ptr) + ((((r109_z / $(8192)) % $(8)) * $(8192)) + (((r109_z / $(64)) % $(128)) * $(64)) + (r109_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_3_ptr) + ((((r109_z / $(8192)) % $(8)) * $(8192)) + (((r109_z / $(64)) % $(128)) * $(64)) + (r109_z % $(64))), dtype=tl.bfloat16))
    r109_value = r109_value.to(tl.bfloat16)
    r109_value = tl.broadcast_to(r109_value, [$(65536)])
    tl.store($(vadd_12_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(262144))
    r110_value = tl.where((r110_z % $(256)) < $(128), tl.load($(vbmm_4_ptr) + ((((((((r110_z / $(32768)) % $(8)) * $(24576)) + (((r110_z % $(256)) - $(0)) * $(128)) + ((r110_z / $(256)) % $(128))) / $(24576)) % $(8)) * $(24576)) + ((((((r110_z / $(32768)) % $(8)) * $(192)) + ((r110_z % $(256)) - $(0))) % $(192)) * $(128)) + ((r110_z / $(256)) % $(128))), dtype=tl.bfloat16), tl.load($(vbmm_2_ptr) + ((((((((r110_z / $(32768)) % $(8)) * $(16384)) + (((r110_z / $(256)) % $(128)) * $(128)) + ((r110_z % $(256)) - $(128))) / $(16384)) % $(8)) * $(16384)) + (((((((r110_z / $(32768)) % $(8)) * $(16384)) + (((r110_z / $(256)) % $(128)) * $(128)) + ((r110_z % $(256)) - $(128))) / $(128)) % $(128)) * $(128)) + (((((r110_z / $(32768)) % $(8)) * $(16384)) + (((r110_z / $(256)) % $(128)) * $(128)) + ((r110_z % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16))
    r110_value = r110_value.to(tl.bfloat16)
    r110_value = tl.broadcast_to(r110_value, [$(262144)])
    tl.store($(vcat_8_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(196608))
    r111_value = tl.where((r111_z % $(192)) < $(128), tl.load($(vbmm_5_ptr) + ((((((((r111_z / $(24576)) % $(8)) * $(24576)) + (((r111_z / $(192)) % $(128)) * $(192)) + ((r111_z % $(192)) - $(0))) / $(24576)) % $(8)) * $(24576)) + (((((((r111_z / $(24576)) % $(8)) * $(24576)) + (((r111_z / $(192)) % $(128)) * $(192)) + ((r111_z % $(192)) - $(0))) / $(192)) % $(128)) * $(192)) + (((((r111_z / $(24576)) % $(8)) * $(24576)) + (((r111_z / $(192)) % $(128)) * $(192)) + ((r111_z % $(192)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(vadd_12_ptr) + ((((r111_z / $(24576)) % $(8)) * $(8192)) + (((r111_z / $(192)) % $(128)) * $(64)) + (((r111_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(196608)])
    tl.store($(vcat_9_ptr) + r111_z, r111_value)
    r112_row = tl.arange($(128))
    r112_col = tl.arange($(512))
    r112_lane = tl.arange($(2048))
    r112_left = tl.load($(vcat_8_ptr) + ((((((((((r112_row[:, None] * $(2048)) + r112_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r112_row[:, None] * $(2048)) + r112_lane[None, :]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r112_row[:, None] * $(2048)) + r112_lane[None, :]) / $(2048)) % $(128)) * $(256)) + (((((((r112_row[:, None] * $(2048)) + r112_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r112_row[:, None] * $(2048)) + r112_lane[None, :]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r112_right = tl.load($(vto_copy_5_ptr) + (((r112_lane[:, None] % $(2048)) * $(512)) + (r112_col[None, :] % $(512))), dtype=tl.bfloat16)
    r112_value = tl.dot(r112_left, r112_right)
    r112_value = r112_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r112_row[:, None] * $(512)) + r112_col[None, :]), r112_value)
    r113_z = tl.arange($(65536))
    r113_value = tl.load($(vmm_6_ptr) + ((((r113_z / $(512)) % $(128)) * $(512)) + (r113_z % $(512))), dtype=tl.bfloat16)
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(65536)])
    tl.store($(vto_copy_26_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(65536))
    r114_value = (tl.load($(vto_copy_26_ptr) + ((((r114_z / $(512)) % $(128)) * $(512)) + (r114_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_4_ptr) + (r114_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(65536)])
    tl.store($(vmul_31_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(65536))
    r115_value = (tl.load($(vmul_31_ptr) + ((((r115_z / $(512)) % $(128)) * $(512)) + (r115_z % $(512))), dtype=tl.float32) * tl.load($(vto_copy_20_ptr) + ((((r115_z / $(512)) % $(128)) * $(512)) + (r115_z % $(512))), dtype=tl.float32))
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(65536)])
    tl.store($(vmul_32_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(65536))
    r116_value = (tl.load($(vmul_31_ptr) + ((((r116_z / $(512)) % $(128)) * $(512)) + (r116_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r116_z / $(512)) % $(128)), dtype=tl.float32))
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(65536)])
    tl.store($(vmul_33_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(128))
    r117_lane = tl.arange($(512))
    r117_value = tl.sum(tl.load($(vmul_32_ptr) + ((r117_z[:, None] * $(512)) + r117_lane[None, :]), dtype=tl.float32), axis=1)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(128)])
    tl.store($(vsum_2_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(128))
    r118_value = ((tl.load($(vrsqrt_1_ptr) + r118_z, dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + r118_z, dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + r118_z, dtype=tl.float32)))
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(128)])
    tl.store($(vpow_4_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(128))
    r119_value = (tl.load($(vsum_2_ptr) + r119_z, dtype=tl.float32) * -0.5)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(128)])
    tl.store($(vmul_34_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(128))
    r120_value = (tl.load($(vmul_34_ptr) + r120_z, dtype=tl.float32) * tl.load($(vpow_4_ptr) + r120_z, dtype=tl.float32))
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(128)])
    tl.store($(vmul_35_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(65536))
    r121_value = (tl.load($(vmul_35_ptr) + ((r121_z / $(512)) % $(128)), dtype=tl.float32) / 512.0)
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(65536)])
    tl.store($(vdiv_1_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(65536))
    r122_value = ((tl.load($(vto_copy_20_ptr) + ((((r122_z / $(512)) % $(128)) * $(512)) + (r122_z % $(512))), dtype=tl.float32)))
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(65536)])
    tl.store($(vpow_5_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(65536))
    r123_value = (tl.load($(vpow_5_ptr) + ((((r123_z / $(512)) % $(128)) * $(512)) + (r123_z % $(512))), dtype=tl.float32) * 2.0)
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(65536)])
    tl.store($(vmul_36_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(65536))
    r124_value = (tl.load($(vdiv_1_ptr) + ((((r124_z / $(512)) % $(128)) * $(512)) + (r124_z % $(512))), dtype=tl.float32) * tl.load($(vmul_36_ptr) + ((((r124_z / $(512)) % $(128)) * $(512)) + (r124_z % $(512))), dtype=tl.float32))
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(65536)])
    tl.store($(vmul_37_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(65536))
    r125_value = (tl.load($(vmul_33_ptr) + ((((r125_z / $(512)) % $(128)) * $(512)) + (r125_z % $(512))), dtype=tl.float32) + tl.load($(vmul_37_ptr) + ((((r125_z / $(512)) % $(128)) * $(512)) + (r125_z % $(512))), dtype=tl.float32))
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(65536)])
    tl.store($(vadd_13_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(65536))
    r126_value = tl.load($(vadd_13_ptr) + ((((r126_z / $(512)) % $(128)) * $(512)) + (r126_z % $(512))), dtype=tl.float32)
    r126_value = r126_value.to(tl.bfloat16)
    r126_value = tl.broadcast_to(r126_value, [$(65536)])
    tl.store($(vto_copy_27_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(73728))
    r127_value = tl.where((r127_z % $(576)) < $(512), tl.load($(vto_copy_27_ptr) + ((((r127_z / $(576)) % $(128)) * $(512)) + (((r127_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(vadd_11_ptr) + ((((((((r127_z / $(576)) % $(128)) * $(64)) + ((r127_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r127_z / $(576)) % $(128)) * $(64)) + ((r127_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r127_value = r127_value.to(tl.bfloat16)
    r127_value = tl.broadcast_to(r127_value, [$(73728)])
    tl.store($(vcat_10_ptr) + r127_z, r127_value)
    r128_row = tl.arange($(128))
    r128_col = tl.arange($(512))
    r128_lane = tl.arange($(576))
    r128_left = tl.load($(vcat_10_ptr) + ((((((r128_row[:, None] * $(576)) + r128_lane[None, :]) / $(576)) % $(128)) * $(576)) + (((r128_row[:, None] * $(576)) + r128_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r128_right = tl.load($(vto_copy_3_ptr) + (((r128_lane[:, None] % $(576)) * $(512)) + (r128_col[None, :] % $(512))), dtype=tl.bfloat16)
    r128_value = tl.dot(r128_left, r128_right)
    r128_value = r128_value.to(tl.bfloat16)
    tl.store($(vmm_7_ptr) + ((r128_row[:, None] * $(512)) + r128_col[None, :]), r128_value)
    r129_row = tl.arange($(128))
    r129_col = tl.arange($(256))
    r129_lane = tl.arange($(1536))
    r129_left = tl.load($(vcat_9_ptr) + ((((((((((r129_row[:, None] * $(1536)) + r129_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r129_row[:, None] * $(1536)) + r129_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r129_row[:, None] * $(1536)) + r129_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r129_row[:, None] * $(1536)) + r129_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r129_row[:, None] * $(1536)) + r129_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r129_right = tl.load($(vto_copy_2_ptr) + (((r129_lane[:, None] % $(1536)) * $(256)) + (r129_col[None, :] % $(256))), dtype=tl.bfloat16)
    r129_value = tl.dot(r129_left, r129_right)
    r129_value = r129_value.to(tl.bfloat16)
    tl.store($(vmm_8_ptr) + ((r129_row[:, None] * $(256)) + r129_col[None, :]), r129_value)
    r130_z = tl.arange($(32768))
    r130_value = tl.load($(vmm_8_ptr) + ((((r130_z / $(256)) % $(128)) * $(256)) + (r130_z % $(256))), dtype=tl.bfloat16)
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(32768)])
    tl.store($(vto_copy_28_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(32768))
    r131_value = (tl.load($(vto_copy_28_ptr) + ((((r131_z / $(256)) % $(128)) * $(256)) + (r131_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_1_ptr) + (r131_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(32768)])
    tl.store($(vmul_38_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(32768))
    r132_value = (tl.load($(vmul_38_ptr) + ((((r132_z / $(256)) % $(128)) * $(256)) + (r132_z % $(256))), dtype=tl.float32) * tl.load($(vto_copy_18_ptr) + ((((r132_z / $(256)) % $(128)) * $(256)) + (r132_z % $(256))), dtype=tl.float32))
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(32768)])
    tl.store($(vmul_39_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(32768))
    r133_value = (tl.load($(vmul_38_ptr) + ((((r133_z / $(256)) % $(128)) * $(256)) + (r133_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((r133_z / $(256)) % $(128)), dtype=tl.float32))
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(32768)])
    tl.store($(vmul_40_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(128))
    r134_lane = tl.arange($(256))
    r134_value = tl.sum(tl.load($(vmul_39_ptr) + ((r134_z[:, None] * $(256)) + r134_lane[None, :]), dtype=tl.float32), axis=1)
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(128)])
    tl.store($(vsum_3_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(128))
    r135_value = ((tl.load($(vrsqrt_ptr) + r135_z, dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + r135_z, dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + r135_z, dtype=tl.float32)))
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(128)])
    tl.store($(vpow_6_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(128))
    r136_value = (tl.load($(vsum_3_ptr) + r136_z, dtype=tl.float32) * -0.5)
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(128)])
    tl.store($(vmul_41_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(128))
    r137_value = (tl.load($(vmul_41_ptr) + r137_z, dtype=tl.float32) * tl.load($(vpow_6_ptr) + r137_z, dtype=tl.float32))
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(128)])
    tl.store($(vmul_42_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(32768))
    r138_value = (tl.load($(vmul_42_ptr) + ((r138_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(32768)])
    tl.store($(vdiv_2_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(32768))
    r139_value = ((tl.load($(vto_copy_18_ptr) + ((((r139_z / $(256)) % $(128)) * $(256)) + (r139_z % $(256))), dtype=tl.float32)))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(32768)])
    tl.store($(vpow_7_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(32768))
    r140_value = (tl.load($(vpow_7_ptr) + ((((r140_z / $(256)) % $(128)) * $(256)) + (r140_z % $(256))), dtype=tl.float32) * 2.0)
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(32768)])
    tl.store($(vmul_43_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(32768))
    r141_value = (tl.load($(vdiv_2_ptr) + ((((r141_z / $(256)) % $(128)) * $(256)) + (r141_z % $(256))), dtype=tl.float32) * tl.load($(vmul_43_ptr) + ((((r141_z / $(256)) % $(128)) * $(256)) + (r141_z % $(256))), dtype=tl.float32))
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(32768)])
    tl.store($(vmul_44_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(32768))
    r142_value = (tl.load($(vmul_40_ptr) + ((((r142_z / $(256)) % $(128)) * $(256)) + (r142_z % $(256))), dtype=tl.float32) + tl.load($(vmul_44_ptr) + ((((r142_z / $(256)) % $(128)) * $(256)) + (r142_z % $(256))), dtype=tl.float32))
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(32768)])
    tl.store($(vadd_14_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(32768))
    r143_value = tl.load($(vadd_14_ptr) + ((((r143_z / $(256)) % $(128)) * $(256)) + (r143_z % $(256))), dtype=tl.float32)
    r143_value = r143_value.to(tl.bfloat16)
    r143_value = tl.broadcast_to(r143_value, [$(32768)])
    tl.store($(vto_copy_29_ptr) + r143_z, r143_value)
    r144_row = tl.arange($(128))
    r144_col = tl.arange($(512))
    r144_lane = tl.arange($(256))
    r144_left = tl.load($(vto_copy_29_ptr) + ((((((r144_row[:, None] * $(256)) + r144_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r144_row[:, None] * $(256)) + r144_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r144_right = tl.load($(vto_copy_ptr) + (((r144_lane[:, None] % $(256)) * $(512)) + (r144_col[None, :] % $(512))), dtype=tl.bfloat16)
    r144_value = tl.dot(r144_left, r144_right)
    r144_value = r144_value.to(tl.bfloat16)
    tl.store($(vmm_9_ptr) + ((r144_row[:, None] * $(512)) + r144_col[None, :]), r144_value)
    r145_z = tl.arange($(65536))
    r145_value = (tl.load($(vmm_7_ptr) + ((((r145_z / $(512)) % $(128)) * $(512)) + (r145_z % $(512))), dtype=tl.bfloat16) + tl.load($(vmm_9_ptr) + ((((r145_z / $(512)) % $(128)) * $(512)) + (r145_z % $(512))), dtype=tl.bfloat16))
    r145_value = r145_value.to(tl.bfloat16)
    r145_value = tl.broadcast_to(r145_value, [$(65536)])
    tl.store($(vadd_15_ptr) + r145_z, r145_value)
    z = tl.arange($(65536))
    value = tl.load($(vadd_15_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
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
  let vsub_4_ptr : RegionName := "vsub_4"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_5_ptr : RegionName := "vsub_5"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
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
  let vcat_8_ptr : RegionName := "vcat_8"
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
  let vcat_10_ptr : RegionName := "vcat_10"
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
  let candidate_vsub_4_ptr : RegionName := "candidate_vsub_4"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vcat_4_ptr : RegionName := "candidate_vcat_4"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vsub_5_ptr : RegionName := "candidate_vsub_5"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vcat_5_ptr : RegionName := "candidate_vcat_5"
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
  let candidate_vsub_6_ptr : RegionName := "candidate_vsub_6"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vlogsumexp_ptr : RegionName := "candidate_vlogsumexp"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_7_ptr : RegionName := "candidate_vsub_7"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vlogsumexp_1_ptr : RegionName := "candidate_vlogsumexp_1"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vsub_8_ptr : RegionName := "candidate_vsub_8"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vmm_8_ptr : RegionName := "candidate_vmm_8"
  let candidate_vlogsumexp_2_ptr : RegionName := "candidate_vlogsumexp_2"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmm_10_ptr : RegionName := "candidate_vmm_10"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vsub_9_ptr : RegionName := "candidate_vsub_9"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsub_10_ptr : RegionName := "candidate_vsub_10"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_12_ptr : RegionName := "candidate_vsub_12"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsub_13_ptr : RegionName := "candidate_vsub_13"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vsub_14_ptr : RegionName := "candidate_vsub_14"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vto_copy_56_ptr : RegionName := "candidate_vto_copy_56"
  let candidate_vto_copy_57_ptr : RegionName := "candidate_vto_copy_57"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vexp_2_ptr : RegionName := "candidate_vexp_2"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vto_copy_58_ptr : RegionName := "candidate_vto_copy_58"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vto_copy_59_ptr : RegionName := "candidate_vto_copy_59"
  let candidate_vbmm_21_ptr : RegionName := "candidate_vbmm_21"
  let candidate_vto_copy_60_ptr : RegionName := "candidate_vto_copy_60"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vbmm_23_ptr : RegionName := "candidate_vbmm_23"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vcat_14_ptr : RegionName := "candidate_vcat_14"
  let candidate_vto_copy_61_ptr : RegionName := "candidate_vto_copy_61"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vpow_4_ptr : RegionName := "candidate_vpow_4"
  let candidate_vmul_50_ptr : RegionName := "candidate_vmul_50"
  let candidate_vmul_51_ptr : RegionName := "candidate_vmul_51"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_vpow_5_ptr : RegionName := "candidate_vpow_5"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vto_copy_62_ptr : RegionName := "candidate_vto_copy_62"
  let candidate_vcat_15_ptr : RegionName := "candidate_vcat_15"
  let candidate_vmm_11_ptr : RegionName := "candidate_vmm_11"
  let candidate_vmm_12_ptr : RegionName := "candidate_vmm_12"
  let candidate_vto_copy_63_ptr : RegionName := "candidate_vto_copy_63"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vmul_56_ptr : RegionName := "candidate_vmul_56"
  let candidate_vsum_6_ptr : RegionName := "candidate_vsum_6"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_57_ptr : RegionName := "candidate_vmul_57"
  let candidate_vmul_58_ptr : RegionName := "candidate_vmul_58"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vpow_7_ptr : RegionName := "candidate_vpow_7"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
  let candidate_vto_copy_64_ptr : RegionName := "candidate_vto_copy_64"
  let candidate_vmm_13_ptr : RegionName := "candidate_vmm_13"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
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
    r21_z = tl.arange($(4096))
    r21_value = (tl.cast(tl.toReal(((r21_z / $(32)) % $(128))), tl.float32) * tl.load($(candidate_vmul_7_ptr) + (r21_z % $(32)), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(4096)])
    tl.store($(candidate_vmul_8_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(8192))
    r22_value = tl.where((r22_z % $(64)) < $(32), tl.load($(candidate_vmul_8_ptr) + ((((r22_z / $(64)) % $(128)) * $(32)) + (((r22_z % $(64)) - $(0)) % $(32))), dtype=tl.float32), tl.load($(candidate_vmul_8_ptr) + ((((r22_z / $(64)) % $(128)) * $(32)) + (((r22_z % $(64)) - $(32)) % $(32))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(8192)])
    tl.store($(candidate_vcat_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(8192))
    r23_value = tl.cos(tl.load($(candidate_vcat_1_ptr) + ((((r23_z / $(64)) % $(128)) * $(64)) + (r23_z % $(64))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(8192)])
    tl.store($(candidate_vcos_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(8192))
    r24_value = tl.load($(candidate_vcos_ptr) + (((((((tl.load($(candidate_vcat_ptr) + ((r24_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r24_z % $(64))) / $(64)) % $(128)) % $(128)) * $(64)) + ((((tl.load($(candidate_vcat_ptr) + ((r24_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r24_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(8192)])
    tl.store($(candidate_vindex_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(8192))
    r25_value = tl.sin(tl.load($(candidate_vcat_1_ptr) + ((((r25_z / $(64)) % $(128)) * $(64)) + (r25_z % $(64))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(8192)])
    tl.store($(candidate_vsin_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(8192))
    r26_value = tl.load($(candidate_vsin_ptr) + (((((((tl.load($(candidate_vcat_ptr) + ((r26_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r26_z % $(64))) / $(64)) % $(128)) % $(128)) * $(64)) + ((((tl.load($(candidate_vcat_ptr) + ((r26_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r26_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(8192)])
    tl.store($(candidate_vindex_1_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(8192))
    r27_value = tl.load($(candidate_vindex_ptr) + ((((r27_z / $(64)) % $(128)) * $(64)) + (r27_z % $(64))), dtype=tl.float32)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(8192)])
    tl.store($(candidate_vto_copy_14_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(8192))
    r28_value = tl.load($(candidate_vindex_1_ptr) + ((((r28_z / $(64)) % $(128)) * $(64)) + (r28_z % $(64))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(8192)])
    tl.store($(candidate_vto_copy_15_ptr) + r28_z, r28_value)
    r29_row = tl.arange($(128))
    r29_col = tl.arange($(256))
    r29_lane = tl.arange($(512))
    r29_left = tl.load($(input0_ptr) + ((((((r29_row[:, None] * $(512)) + r29_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r29_row[:, None] * $(512)) + r29_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r29_right = tl.load($(candidate_vto_copy_ptr) + (((r29_col[None, :] % $(256)) * $(512)) + (r29_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r29_value = tl.dot(r29_left, r29_right)
    r29_value = r29_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r29_row[:, None] * $(256)) + r29_col[None, :]), r29_value)
    r30_z = tl.arange($(32768))
    r30_value = tl.load($(candidate_vmm_ptr) + ((((r30_z / $(256)) % $(128)) * $(256)) + (r30_z % $(256))), dtype=tl.bfloat16)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(32768)])
    tl.store($(candidate_vto_copy_18_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(32768))
    r31_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r31_z / $(256)) % $(128)) * $(256)) + (r31_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_18_ptr) + ((((r31_z / $(256)) % $(128)) * $(256)) + (r31_z % $(256))), dtype=tl.float32)))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(32768)])
    tl.store($(candidate_vpow_2_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(128))
    r32_lane = tl.arange($(256))
    r32_value = (tl.sum(tl.load($(candidate_vpow_2_ptr) + ((r32_z[:, None] * $(256)) + r32_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(128)])
    tl.store($(candidate_vmean_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(128))
    r33_value = (tl.load($(candidate_vmean_ptr) + r33_z, dtype=tl.float32) + 0.000001)
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(128)])
    tl.store($(candidate_vadd_2_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(128))
    r34_value = tl.rsqrt(tl.load($(candidate_vadd_2_ptr) + r34_z, dtype=tl.float32))
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(128)])
    tl.store($(candidate_vrsqrt_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(32768))
    r35_value = (tl.load($(candidate_vto_copy_18_ptr) + ((((r35_z / $(256)) % $(128)) * $(256)) + (r35_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r35_z / $(256)) % $(128)), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(32768)])
    tl.store($(candidate_vmul_9_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(32768))
    r36_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r36_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_9_ptr) + ((((r36_z / $(256)) % $(128)) * $(256)) + (r36_z % $(256))), dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(32768)])
    tl.store($(candidate_vmul_10_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(32768))
    r37_value = tl.load($(candidate_vmul_10_ptr) + ((((r37_z / $(256)) % $(128)) * $(256)) + (r37_z % $(256))), dtype=tl.float32)
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(32768)])
    tl.store($(candidate_vto_copy_19_ptr) + r37_z, r37_value)
    r38_row = tl.arange($(128))
    r38_col = tl.arange($(1536))
    r38_lane = tl.arange($(256))
    r38_left = tl.load($(candidate_vto_copy_19_ptr) + ((((((r38_row[:, None] * $(256)) + r38_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r38_row[:, None] * $(256)) + r38_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r38_right = tl.load($(candidate_vto_copy_2_ptr) + (((r38_col[None, :] % $(1536)) * $(256)) + (r38_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r38_value = tl.dot(r38_left, r38_right)
    r38_value = r38_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r38_row[:, None] * $(1536)) + r38_col[None, :]), r38_value)
    r39_row = tl.arange($(128))
    r39_col = tl.arange($(576))
    r39_lane = tl.arange($(512))
    r39_left = tl.load($(input0_ptr) + ((((((r39_row[:, None] * $(512)) + r39_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r39_row[:, None] * $(512)) + r39_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r39_right = tl.load($(candidate_vto_copy_3_ptr) + (((r39_col[None, :] % $(576)) * $(512)) + (r39_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r39_value = tl.dot(r39_left, r39_right)
    r39_value = r39_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r39_row[:, None] * $(576)) + r39_col[None, :]), r39_value)
    r40_z = tl.arange($(65536))
    r40_value = tl.load($(candidate_vmm_2_ptr) + ((((r40_z / $(512)) % $(128)) * $(576)) + (r40_z % $(512))), dtype=tl.bfloat16)
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(candidate_vto_copy_20_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(65536))
    r41_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r41_z / $(512)) % $(128)) * $(512)) + (r41_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_20_ptr) + ((((r41_z / $(512)) % $(128)) * $(512)) + (r41_z % $(512))), dtype=tl.float32)))
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(65536)])
    tl.store($(candidate_vpow_3_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(128))
    r42_lane = tl.arange($(512))
    r42_value = (tl.sum(tl.load($(candidate_vpow_3_ptr) + ((r42_z[:, None] * $(512)) + r42_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(128)])
    tl.store($(candidate_vmean_1_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(128))
    r43_value = (tl.load($(candidate_vmean_1_ptr) + r43_z, dtype=tl.float32) + 0.000001)
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(128)])
    tl.store($(candidate_vadd_3_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(128))
    r44_value = tl.rsqrt(tl.load($(candidate_vadd_3_ptr) + r44_z, dtype=tl.float32))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(128)])
    tl.store($(candidate_vrsqrt_1_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(65536))
    r45_value = (tl.load($(candidate_vto_copy_20_ptr) + ((((r45_z / $(512)) % $(128)) * $(512)) + (r45_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r45_z / $(512)) % $(128)), dtype=tl.float32))
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(65536)])
    tl.store($(candidate_vmul_11_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(65536))
    r46_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r46_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_11_ptr) + ((((r46_z / $(512)) % $(128)) * $(512)) + (r46_z % $(512))), dtype=tl.float32))
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(65536)])
    tl.store($(candidate_vmul_12_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(65536))
    r47_value = tl.load($(candidate_vmul_12_ptr) + ((((r47_z / $(512)) % $(128)) * $(512)) + (r47_z % $(512))), dtype=tl.float32)
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(65536)])
    tl.store($(candidate_vto_copy_21_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(32768))
    r48_value = (tl.load($(candidate_vmm_1_ptr) + ((((r48_z / $(32)) % $(128)) * $(1536)) + (((r48_z / $(4096)) % $(8)) * $(192)) + ((r48_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r48_z / $(32)) % $(128)) * $(64)) + (r48_z % $(32))), dtype=tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(32768)])
    tl.store($(candidate_vmul_13_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(32768))
    r49_value = (tl.load($(candidate_vmm_1_ptr) + ((((r49_z / $(32)) % $(128)) * $(1536)) + (((r49_z / $(4096)) % $(8)) * $(192)) + ((r49_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r49_z / $(32)) % $(128)) * $(64)) + (r49_z % $(32))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(32768)])
    tl.store($(candidate_vmul_14_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(32768))
    r50_value = (tl.load($(candidate_vmul_13_ptr) + ((((r50_z / $(4096)) % $(8)) * $(4096)) + (((r50_z / $(32)) % $(128)) * $(32)) + (r50_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_14_ptr) + ((((r50_z / $(4096)) % $(8)) * $(4096)) + (((r50_z / $(32)) % $(128)) * $(32)) + (r50_z % $(32))), dtype=tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(32768)])
    tl.store($(candidate_vsub_4_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(32768))
    r51_value = (tl.load($(candidate_vmm_1_ptr) + ((((r51_z / $(32)) % $(128)) * $(1536)) + (((r51_z / $(4096)) % $(8)) * $(192)) + ((r51_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r51_z / $(32)) % $(128)) * $(64)) + (r51_z % $(32))), dtype=tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(32768)])
    tl.store($(candidate_vmul_15_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(32768))
    r52_value = (tl.load($(candidate_vmm_1_ptr) + ((((r52_z / $(32)) % $(128)) * $(1536)) + (((r52_z / $(4096)) % $(8)) * $(192)) + ((r52_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r52_z / $(32)) % $(128)) * $(64)) + (r52_z % $(32))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(32768)])
    tl.store($(candidate_vmul_16_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(32768))
    r53_value = (tl.load($(candidate_vmul_15_ptr) + ((((r53_z / $(4096)) % $(8)) * $(4096)) + (((r53_z / $(32)) % $(128)) * $(32)) + (r53_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_16_ptr) + ((((r53_z / $(4096)) % $(8)) * $(4096)) + (((r53_z / $(32)) % $(128)) * $(32)) + (r53_z % $(32))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(32768)])
    tl.store($(candidate_vadd_4_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(65536))
    r54_value = tl.where((r54_z % $(64)) < $(32), tl.load($(candidate_vsub_4_ptr) + ((((r54_z / $(8192)) % $(8)) * $(4096)) + (((r54_z / $(64)) % $(128)) * $(32)) + (((r54_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_4_ptr) + ((((r54_z / $(8192)) % $(8)) * $(4096)) + (((r54_z / $(64)) % $(128)) * $(32)) + (((r54_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(65536)])
    tl.store($(candidate_vcat_4_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(4096))
    r55_value = (tl.load($(candidate_vmm_2_ptr) + ((((r55_z / $(32)) % $(128)) * $(576)) + ((r55_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r55_z / $(32)) % $(128)) * $(64)) + (r55_z % $(32))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(4096)])
    tl.store($(candidate_vmul_17_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(4096))
    r56_value = (tl.load($(candidate_vmm_2_ptr) + ((((r56_z / $(32)) % $(128)) * $(576)) + ((r56_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r56_z / $(32)) % $(128)) * $(64)) + (r56_z % $(32))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(4096)])
    tl.store($(candidate_vmul_18_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(4096))
    r57_value = (tl.load($(candidate_vmul_17_ptr) + ((((r57_z / $(32)) % $(128)) * $(32)) + (r57_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_18_ptr) + ((((r57_z / $(32)) % $(128)) * $(32)) + (r57_z % $(32))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(4096)])
    tl.store($(candidate_vsub_5_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(4096))
    r58_value = (tl.load($(candidate_vmm_2_ptr) + ((((r58_z / $(32)) % $(128)) * $(576)) + ((r58_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r58_z / $(32)) % $(128)) * $(64)) + (r58_z % $(32))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(4096)])
    tl.store($(candidate_vmul_19_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(4096))
    r59_value = (tl.load($(candidate_vmm_2_ptr) + ((((r59_z / $(32)) % $(128)) * $(576)) + ((r59_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r59_z / $(32)) % $(128)) * $(64)) + (r59_z % $(32))), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(4096)])
    tl.store($(candidate_vmul_20_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(4096))
    r60_value = (tl.load($(candidate_vmul_19_ptr) + ((((r60_z / $(32)) % $(128)) * $(32)) + (r60_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_20_ptr) + ((((r60_z / $(32)) % $(128)) * $(32)) + (r60_z % $(32))), dtype=tl.bfloat16))
    r60_value = r60_value.to(tl.bfloat16)
    r60_value = tl.broadcast_to(r60_value, [$(4096)])
    tl.store($(candidate_vadd_5_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(8192))
    r61_value = tl.where((r61_z % $(64)) < $(32), tl.load($(candidate_vsub_5_ptr) + ((((r61_z / $(64)) % $(128)) * $(32)) + (((r61_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_5_ptr) + ((((r61_z / $(64)) % $(128)) * $(32)) + (((r61_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(8192)])
    tl.store($(candidate_vcat_5_ptr) + r61_z, r61_value)
    r62_row = tl.arange($(128))
    r62_col = tl.arange($(512))
    r62_lane = tl.arange($(128))
    for r62_group in range($(8)) {
      r62_left = tl.load($(candidate_vmm_1_ptr) + (((((((r62_group % $(8)) * $(16384)) + (r62_row[:, None] * $(128)) + r62_lane[None, :]) / $(128)) % $(128)) * $(1536)) + ((((((r62_group % $(8)) * $(16384)) + (r62_row[:, None] * $(128)) + r62_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r62_group % $(8)) * $(16384)) + (r62_row[:, None] * $(128)) + r62_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r62_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r62_group % $(8)) * $(65536)) + (r62_lane[:, None] * $(512)) + r62_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r62_group % $(8)) * $(65536)) + (r62_lane[:, None] * $(512)) + r62_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r62_group % $(8)) * $(65536)) + (r62_lane[:, None] * $(512)) + r62_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r62_value = tl.dot(r62_left, r62_right)
      r62_value = r62_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r62_group * $(65536)) + (r62_row[:, None] * $(512)) + r62_col[None, :]), r62_value)
    }
    r63_z = tl.arange($(24576))
    r63_value = tl.load($(candidate_vcat_4_ptr) + ((((r63_z / $(3072)) % $(8)) * $(8192)) + (((r63_z / $(64)) % $(48)) * $(64)) + (r63_z % $(64))), dtype=tl.bfloat16)
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(24576)])
    tl.store($(candidate_vto_copy_22_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(3072))
    r64_value = tl.load($(candidate_vcat_5_ptr) + ((((r64_z / $(64)) % $(48)) * $(64)) + (r64_z % $(64))), dtype=tl.bfloat16)
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(3072)])
    tl.store($(candidate_vto_copy_23_ptr) + r64_z, r64_value)
    r65_row = tl.arange($(48))
    r65_col = tl.arange($(48))
    r65_lane = tl.arange($(64))
    for r65_group in range($(8)) {
      r65_left = tl.load($(candidate_vto_copy_22_ptr) + (((((((r65_group % $(8)) * $(3072)) + (r65_row[:, None] * $(64)) + r65_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r65_group % $(8)) * $(3072)) + (r65_row[:, None] * $(64)) + r65_lane[None, :]) / $(64)) % $(48)) * $(64)) + ((((r65_group % $(8)) * $(3072)) + (r65_row[:, None] * $(64)) + r65_lane[None, :]) % $(64))), dtype=tl.float32)
      r65_right = tl.load($(candidate_vto_copy_23_ptr) + ((((((r65_group % $(8)) * $(3072)) + (r65_lane[:, None] * $(48)) + r65_col[None, :]) % $(48)) * $(64)) + (((((r65_group % $(8)) * $(3072)) + (r65_lane[:, None] * $(48)) + r65_col[None, :]) / $(48)) % $(64))), dtype=tl.float32)
      r65_value = tl.dot(r65_left, r65_right)
      r65_value = r65_value.to(tl.float32)
      tl.store($(candidate_vbmm_1_ptr) + ((r65_group * $(2304)) + (r65_row[:, None] * $(48)) + r65_col[None, :]), r65_value)
    }
    r66_z = tl.arange($(196608))
    r66_value = tl.load($(candidate_vbmm_ptr) + ((((r66_z / $(24576)) % $(8)) * $(65536)) + (((r66_z / $(512)) % $(48)) * $(512)) + (r66_z % $(512))), dtype=tl.bfloat16)
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(196608)])
    tl.store($(candidate_vto_copy_24_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(24576))
    r67_value = tl.load($(candidate_vto_copy_21_ptr) + ((((r67_z / $(512)) % $(48)) * $(512)) + (r67_z % $(512))), dtype=tl.bfloat16)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(24576)])
    tl.store($(candidate_vto_copy_25_ptr) + r67_z, r67_value)
    r68_row = tl.arange($(384))
    r68_col = tl.arange($(48))
    r68_lane = tl.arange($(512))
    r68_left = tl.load($(candidate_vto_copy_24_ptr) + ((((((r68_row[:, None] * $(512)) + r68_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r68_row[:, None] * $(512)) + r68_lane[None, :]) / $(512)) % $(48)) * $(512)) + (((r68_row[:, None] * $(512)) + r68_lane[None, :]) % $(512))), dtype=tl.float32)
    r68_right = tl.load($(candidate_vto_copy_25_ptr) + (((((r68_lane[:, None] * $(48)) + r68_col[None, :]) % $(48)) * $(512)) + ((((r68_lane[:, None] * $(48)) + r68_col[None, :]) / $(48)) % $(512))), dtype=tl.float32)
    r68_value = tl.dot(r68_left, r68_right)
    r68_value = r68_value.to(tl.float32)
    tl.store($(candidate_vmm_3_ptr) + ((r68_row[:, None] * $(48)) + r68_col[None, :]), r68_value)
    r69_z = tl.arange($(18432))
    r69_value = (tl.load($(candidate_vbmm_1_ptr) + ((((r69_z / $(2304)) % $(8)) * $(2304)) + (((r69_z / $(48)) % $(48)) * $(48)) + (r69_z % $(48))), dtype=tl.float32) + tl.load($(candidate_vmm_3_ptr) + ((((r69_z / $(2304)) % $(8)) * $(2304)) + (((r69_z / $(48)) % $(48)) * $(48)) + (r69_z % $(48))), dtype=tl.float32))
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(18432)])
    tl.store($(candidate_vadd_6_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(18432))
    r70_value = (tl.load($(candidate_vadd_6_ptr) + ((((r70_z / $(2304)) % $(8)) * $(2304)) + (((r70_z / $(48)) % $(48)) * $(48)) + (r70_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(18432)])
    tl.store($(candidate_vmul_21_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(48))
    r71_value = (tl.cast(r71_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r71_value = tl.broadcast_to(r71_value, [$(48)])
    tl.store($(candidate_vadd_7_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(48))
    r72_value = (tl.load($(candidate_vadd_7_ptr) + r72_z, dtype=tl.int64) - tl.full([], $(48), dtype=tl.int64))
    r72_value = tl.broadcast_to(r72_value, [$(48)])
    tl.store($(candidate_vsub_6_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(2304))
    r73_value = (tl.cast((r73_z % $(48)), tl.int64) > tl.load($(candidate_vsub_6_ptr) + ((r73_z / $(48)) % $(48)), dtype=tl.int64))
    r73_value = tl.broadcast_to(r73_value, [$(2304)])
    tl.store($(candidate_vgt_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(18432))
    r74_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r74_z / $(48)) % $(48)) * $(48)) + (r74_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_21_ptr) + ((((r74_z / $(2304)) % $(8)) * $(2304)) + (((r74_z / $(48)) % $(48)) * $(48)) + (r74_z % $(48))), dtype=tl.float32))
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(18432))
    r75_lane = tl.arange($(48))
    r75_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r75_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r75_z[:, None] / $(48)) % $(48)) * $(48)) + (r75_lane[None, :] % $(48))), dtype=tl.float32)
    r75_maximum = tl.max(r75_row_values, axis=1, keep_dims=true)
    r75_weights = tl.exp(r75_row_values - r75_maximum)
    r75_total = tl.sum(r75_weights, axis=1)
    r75_maximum = tl.max(r75_row_values, axis=1)
    r75_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r75_z / $(2304)) % $(8)) * $(2304)) + (((r75_z / $(48)) % $(48)) * $(48)) + (r75_z % $(48))), dtype=tl.float32) - r75_maximum) / r75_total
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(18432)])
    tl.store($(candidate_vsoftmax_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(18432))
    r76_value = tl.load($(candidate_vsoftmax_ptr) + ((((r76_z / $(2304)) % $(8)) * $(2304)) + (((r76_z / $(48)) % $(48)) * $(48)) + (r76_z % $(48))), dtype=tl.float32)
    r76_value = r76_value.to(tl.bfloat16)
    r76_value = tl.broadcast_to(r76_value, [$(18432)])
    tl.store($(candidate_vto_copy_26_ptr) + r76_z, r76_value)
    r77_row = tl.arange($(384))
    r77_col = tl.arange($(512))
    r77_lane = tl.arange($(48))
    r77_left = tl.load($(candidate_vto_copy_26_ptr) + ((((((r77_row[:, None] * $(48)) + r77_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r77_row[:, None] * $(48)) + r77_lane[None, :]) / $(48)) % $(48)) * $(48)) + (((r77_row[:, None] * $(48)) + r77_lane[None, :]) % $(48))), dtype=tl.bfloat16)
    r77_right = tl.load($(candidate_vto_copy_21_ptr) + ((((((r77_lane[:, None] * $(512)) + r77_col[None, :]) / $(512)) % $(48)) * $(512)) + (((r77_lane[:, None] * $(512)) + r77_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r77_value = tl.dot(r77_left, r77_right)
    r77_value = r77_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r77_row[:, None] * $(512)) + r77_col[None, :]), r77_value)
    r78_z = tl.arange($(384))
    r78_lane = tl.arange($(48))
    r78_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r78_z[:, None] / $(48)) % $(8)) * $(2304)) + ((r78_z[:, None] % $(48)) * $(48)) + (r78_lane[None, :] % $(48))), dtype=tl.float32)
    r78_maximum = tl.max(r78_row_values, axis=1, keep_dims=true)
    r78_weights = tl.exp(r78_row_values - r78_maximum)
    r78_total = tl.sum(r78_weights, axis=1)
    r78_maximum = tl.max(r78_row_values, axis=1)
    r78_value = r78_maximum + tl.log(r78_total)
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(384)])
    tl.store($(candidate_vlogsumexp_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(8192))
    r79_value = tl.load($(candidate_vcat_4_ptr) + ((((r79_z / $(1024)) % $(8)) * $(8192)) + (((r79_z / $(64)) % $(16)) * $(64)) + (r79_z % $(64)) + $(3072)), dtype=tl.bfloat16)
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(8192)])
    tl.store($(candidate_vto_copy_27_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(1024))
    r80_value = tl.load($(candidate_vcat_5_ptr) + ((((r80_z / $(64)) % $(16)) * $(64)) + (r80_z % $(64)) + $(3072)), dtype=tl.bfloat16)
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(1024)])
    tl.store($(candidate_vto_copy_28_ptr) + r80_z, r80_value)
    r81_row = tl.arange($(16))
    r81_col = tl.arange($(16))
    r81_lane = tl.arange($(64))
    for r81_group in range($(8)) {
      r81_left = tl.load($(candidate_vto_copy_27_ptr) + (((((((r81_group % $(8)) * $(1024)) + (r81_row[:, None] * $(64)) + r81_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r81_group % $(8)) * $(1024)) + (r81_row[:, None] * $(64)) + r81_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r81_group % $(8)) * $(1024)) + (r81_row[:, None] * $(64)) + r81_lane[None, :]) % $(64))), dtype=tl.float32)
      r81_right = tl.load($(candidate_vto_copy_28_ptr) + ((((((r81_group % $(8)) * $(1024)) + (r81_lane[:, None] * $(16)) + r81_col[None, :]) % $(16)) * $(64)) + (((((r81_group % $(8)) * $(1024)) + (r81_lane[:, None] * $(16)) + r81_col[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r81_value = tl.dot(r81_left, r81_right)
      r81_value = r81_value.to(tl.float32)
      tl.store($(candidate_vbmm_2_ptr) + ((r81_group * $(256)) + (r81_row[:, None] * $(16)) + r81_col[None, :]), r81_value)
    }
    r82_z = tl.arange($(65536))
    r82_value = tl.load($(candidate_vbmm_ptr) + ((((r82_z / $(8192)) % $(8)) * $(65536)) + (((r82_z / $(512)) % $(16)) * $(512)) + (r82_z % $(512)) + $(24576)), dtype=tl.bfloat16)
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(65536)])
    tl.store($(candidate_vto_copy_29_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(8192))
    r83_value = tl.load($(candidate_vto_copy_21_ptr) + ((((r83_z / $(512)) % $(16)) * $(512)) + (r83_z % $(512)) + $(24576)), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(8192)])
    tl.store($(candidate_vto_copy_30_ptr) + r83_z, r83_value)
    r84_row = tl.arange($(128))
    r84_col = tl.arange($(16))
    r84_lane = tl.arange($(512))
    r84_left = tl.load($(candidate_vto_copy_29_ptr) + ((((((r84_row[:, None] * $(512)) + r84_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r84_row[:, None] * $(512)) + r84_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r84_row[:, None] * $(512)) + r84_lane[None, :]) % $(512))), dtype=tl.float32)
    r84_right = tl.load($(candidate_vto_copy_30_ptr) + (((((r84_lane[:, None] * $(16)) + r84_col[None, :]) % $(16)) * $(512)) + ((((r84_lane[:, None] * $(16)) + r84_col[None, :]) / $(16)) % $(512))), dtype=tl.float32)
    r84_value = tl.dot(r84_left, r84_right)
    r84_value = r84_value.to(tl.float32)
    tl.store($(candidate_vmm_5_ptr) + ((r84_row[:, None] * $(16)) + r84_col[None, :]), r84_value)
    r85_z = tl.arange($(2048))
    r85_value = (tl.load($(candidate_vbmm_2_ptr) + ((((r85_z / $(256)) % $(8)) * $(256)) + (((r85_z / $(16)) % $(16)) * $(16)) + (r85_z % $(16))), dtype=tl.float32) + tl.load($(candidate_vmm_5_ptr) + ((((r85_z / $(256)) % $(8)) * $(256)) + (((r85_z / $(16)) % $(16)) * $(16)) + (r85_z % $(16))), dtype=tl.float32))
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(2048)])
    tl.store($(candidate_vadd_8_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(2048))
    r86_value = (tl.load($(candidate_vadd_8_ptr) + ((((r86_z / $(256)) % $(8)) * $(256)) + (((r86_z / $(16)) % $(16)) * $(16)) + (r86_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(2048)])
    tl.store($(candidate_vmul_22_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(16))
    r87_value = (tl.cast(r87_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r87_value = tl.broadcast_to(r87_value, [$(16)])
    tl.store($(candidate_vadd_9_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(16))
    r88_value = (tl.load($(candidate_vadd_9_ptr) + r88_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r88_value = tl.broadcast_to(r88_value, [$(16)])
    tl.store($(candidate_vsub_7_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(256))
    r89_value = (tl.cast((r89_z % $(16)), tl.int64) > tl.load($(candidate_vsub_7_ptr) + ((r89_z / $(16)) % $(16)), dtype=tl.int64))
    r89_value = tl.broadcast_to(r89_value, [$(256)])
    tl.store($(candidate_vgt_1_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(2048))
    r90_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r90_z / $(16)) % $(16)) * $(16)) + (r90_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_22_ptr) + ((((r90_z / $(256)) % $(8)) * $(256)) + (((r90_z / $(16)) % $(16)) * $(16)) + (r90_z % $(16))), dtype=tl.float32))
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(2048))
    r91_lane = tl.arange($(16))
    r91_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r91_z[:, None] / $(256)) % $(8)) * $(256)) + (((r91_z[:, None] / $(16)) % $(16)) * $(16)) + (r91_lane[None, :] % $(16))), dtype=tl.float32)
    r91_maximum = tl.max(r91_row_values, axis=1, keep_dims=true)
    r91_weights = tl.exp(r91_row_values - r91_maximum)
    r91_total = tl.sum(r91_weights, axis=1)
    r91_maximum = tl.max(r91_row_values, axis=1)
    r91_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r91_z / $(256)) % $(8)) * $(256)) + (((r91_z / $(16)) % $(16)) * $(16)) + (r91_z % $(16))), dtype=tl.float32) - r91_maximum) / r91_total
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(2048)])
    tl.store($(candidate_vsoftmax_1_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(2048))
    r92_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r92_z / $(256)) % $(8)) * $(256)) + (((r92_z / $(16)) % $(16)) * $(16)) + (r92_z % $(16))), dtype=tl.float32)
    r92_value = r92_value.to(tl.bfloat16)
    r92_value = tl.broadcast_to(r92_value, [$(2048)])
    tl.store($(candidate_vto_copy_31_ptr) + r92_z, r92_value)
    r93_row = tl.arange($(128))
    r93_col = tl.arange($(512))
    r93_lane = tl.arange($(16))
    r93_left = tl.load($(candidate_vto_copy_31_ptr) + ((((((r93_row[:, None] * $(16)) + r93_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r93_row[:, None] * $(16)) + r93_lane[None, :]) / $(16)) % $(16)) * $(16)) + (((r93_row[:, None] * $(16)) + r93_lane[None, :]) % $(16))), dtype=tl.bfloat16)
    r93_right = tl.load($(candidate_vto_copy_21_ptr) + ((((((r93_lane[:, None] * $(512)) + r93_col[None, :]) / $(512)) % $(16)) * $(512)) + (((r93_lane[:, None] * $(512)) + r93_col[None, :]) % $(512)) + $(24576)), dtype=tl.bfloat16)
    r93_value = tl.dot(r93_left, r93_right)
    r93_value = r93_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r93_row[:, None] * $(512)) + r93_col[None, :]), r93_value)
    r94_z = tl.arange($(128))
    r94_lane = tl.arange($(16))
    r94_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r94_z[:, None] / $(16)) % $(8)) * $(256)) + ((r94_z[:, None] % $(16)) * $(16)) + (r94_lane[None, :] % $(16))), dtype=tl.float32)
    r94_maximum = tl.max(r94_row_values, axis=1, keep_dims=true)
    r94_weights = tl.exp(r94_row_values - r94_maximum)
    r94_total = tl.sum(r94_weights, axis=1)
    r94_maximum = tl.max(r94_row_values, axis=1)
    r94_value = r94_maximum + tl.log(r94_total)
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(128)])
    tl.store($(candidate_vlogsumexp_1_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(32768))
    r95_value = tl.load($(candidate_vcat_4_ptr) + ((((r95_z / $(4096)) % $(8)) * $(8192)) + (((r95_z / $(64)) % $(64)) * $(64)) + (r95_z % $(64)) + $(4096)), dtype=tl.bfloat16)
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(32768)])
    tl.store($(candidate_vto_copy_32_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(4096))
    r96_value = tl.load($(candidate_vcat_5_ptr) + ((((r96_z / $(64)) % $(64)) * $(64)) + (r96_z % $(64)) + $(4096)), dtype=tl.bfloat16)
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(4096)])
    tl.store($(candidate_vto_copy_33_ptr) + r96_z, r96_value)
    r97_row = tl.arange($(64))
    r97_col = tl.arange($(64))
    r97_lane = tl.arange($(64))
    for r97_group in range($(8)) {
      r97_left = tl.load($(candidate_vto_copy_32_ptr) + (((((((r97_group % $(8)) * $(4096)) + (r97_row[:, None] * $(64)) + r97_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r97_group % $(8)) * $(4096)) + (r97_row[:, None] * $(64)) + r97_lane[None, :]) / $(64)) % $(64)) * $(64)) + ((((r97_group % $(8)) * $(4096)) + (r97_row[:, None] * $(64)) + r97_lane[None, :]) % $(64))), dtype=tl.float32)
      r97_right = tl.load($(candidate_vto_copy_33_ptr) + ((((((r97_group % $(8)) * $(4096)) + (r97_lane[:, None] * $(64)) + r97_col[None, :]) % $(64)) * $(64)) + (((((r97_group % $(8)) * $(4096)) + (r97_lane[:, None] * $(64)) + r97_col[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r97_value = tl.dot(r97_left, r97_right)
      r97_value = r97_value.to(tl.float32)
      tl.store($(candidate_vbmm_3_ptr) + ((r97_group * $(4096)) + (r97_row[:, None] * $(64)) + r97_col[None, :]), r97_value)
    }
    r98_z = tl.arange($(262144))
    r98_value = tl.load($(candidate_vbmm_ptr) + ((((r98_z / $(32768)) % $(8)) * $(65536)) + (((r98_z / $(512)) % $(64)) * $(512)) + (r98_z % $(512)) + $(32768)), dtype=tl.bfloat16)
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(262144)])
    tl.store($(candidate_vto_copy_34_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(32768))
    r99_value = tl.load($(candidate_vto_copy_21_ptr) + ((((r99_z / $(512)) % $(64)) * $(512)) + (r99_z % $(512)) + $(32768)), dtype=tl.bfloat16)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(32768)])
    tl.store($(candidate_vto_copy_35_ptr) + r99_z, r99_value)
    r100_row = tl.arange($(512))
    r100_col = tl.arange($(64))
    r100_lane = tl.arange($(512))
    r100_left = tl.load($(candidate_vto_copy_34_ptr) + ((((((r100_row[:, None] * $(512)) + r100_lane[None, :]) / $(32768)) % $(8)) * $(32768)) + (((((r100_row[:, None] * $(512)) + r100_lane[None, :]) / $(512)) % $(64)) * $(512)) + (((r100_row[:, None] * $(512)) + r100_lane[None, :]) % $(512))), dtype=tl.float32)
    r100_right = tl.load($(candidate_vto_copy_35_ptr) + (((((r100_lane[:, None] * $(64)) + r100_col[None, :]) % $(64)) * $(512)) + ((((r100_lane[:, None] * $(64)) + r100_col[None, :]) / $(64)) % $(512))), dtype=tl.float32)
    r100_value = tl.dot(r100_left, r100_right)
    r100_value = r100_value.to(tl.float32)
    tl.store($(candidate_vmm_7_ptr) + ((r100_row[:, None] * $(64)) + r100_col[None, :]), r100_value)
    r101_z = tl.arange($(32768))
    r101_value = (tl.load($(candidate_vbmm_3_ptr) + ((((r101_z / $(4096)) % $(8)) * $(4096)) + (((r101_z / $(64)) % $(64)) * $(64)) + (r101_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_7_ptr) + ((((r101_z / $(4096)) % $(8)) * $(4096)) + (((r101_z / $(64)) % $(64)) * $(64)) + (r101_z % $(64))), dtype=tl.float32))
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(32768)])
    tl.store($(candidate_vadd_10_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(32768))
    r102_value = (tl.load($(candidate_vadd_10_ptr) + ((((r102_z / $(4096)) % $(8)) * $(4096)) + (((r102_z / $(64)) % $(64)) * $(64)) + (r102_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(32768)])
    tl.store($(candidate_vmul_23_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(64))
    r103_value = (tl.cast(r103_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r103_value = tl.broadcast_to(r103_value, [$(64)])
    tl.store($(candidate_vadd_11_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(64))
    r104_value = (tl.load($(candidate_vadd_11_ptr) + r104_z, dtype=tl.int64) - tl.full([], $(64), dtype=tl.int64))
    r104_value = tl.broadcast_to(r104_value, [$(64)])
    tl.store($(candidate_vsub_8_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(4096))
    r105_value = (tl.cast((r105_z % $(64)), tl.int64) > tl.load($(candidate_vsub_8_ptr) + ((r105_z / $(64)) % $(64)), dtype=tl.int64))
    r105_value = tl.broadcast_to(r105_value, [$(4096)])
    tl.store($(candidate_vgt_2_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(32768))
    r106_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r106_z / $(64)) % $(64)) * $(64)) + (r106_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r106_z / $(4096)) % $(8)) * $(4096)) + (((r106_z / $(64)) % $(64)) * $(64)) + (r106_z % $(64))), dtype=tl.float32))
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(32768))
    r107_lane = tl.arange($(64))
    r107_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r107_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r107_z[:, None] / $(64)) % $(64)) * $(64)) + (r107_lane[None, :] % $(64))), dtype=tl.float32)
    r107_maximum = tl.max(r107_row_values, axis=1, keep_dims=true)
    r107_weights = tl.exp(r107_row_values - r107_maximum)
    r107_total = tl.sum(r107_weights, axis=1)
    r107_maximum = tl.max(r107_row_values, axis=1)
    r107_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r107_z / $(4096)) % $(8)) * $(4096)) + (((r107_z / $(64)) % $(64)) * $(64)) + (r107_z % $(64))), dtype=tl.float32) - r107_maximum) / r107_total
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(32768)])
    tl.store($(candidate_vsoftmax_2_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(32768))
    r108_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r108_z / $(4096)) % $(8)) * $(4096)) + (((r108_z / $(64)) % $(64)) * $(64)) + (r108_z % $(64))), dtype=tl.float32)
    r108_value = r108_value.to(tl.bfloat16)
    r108_value = tl.broadcast_to(r108_value, [$(32768)])
    tl.store($(candidate_vto_copy_36_ptr) + r108_z, r108_value)
    r109_row = tl.arange($(512))
    r109_col = tl.arange($(512))
    r109_lane = tl.arange($(64))
    r109_left = tl.load($(candidate_vto_copy_36_ptr) + ((((((r109_row[:, None] * $(64)) + r109_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r109_row[:, None] * $(64)) + r109_lane[None, :]) / $(64)) % $(64)) * $(64)) + (((r109_row[:, None] * $(64)) + r109_lane[None, :]) % $(64))), dtype=tl.bfloat16)
    r109_right = tl.load($(candidate_vto_copy_21_ptr) + ((((((r109_lane[:, None] * $(512)) + r109_col[None, :]) / $(512)) % $(64)) * $(512)) + (((r109_lane[:, None] * $(512)) + r109_col[None, :]) % $(512)) + $(32768)), dtype=tl.bfloat16)
    r109_value = tl.dot(r109_left, r109_right)
    r109_value = r109_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_8_ptr) + ((r109_row[:, None] * $(512)) + r109_col[None, :]), r109_value)
    r110_z = tl.arange($(512))
    r110_lane = tl.arange($(64))
    r110_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r110_z[:, None] / $(64)) % $(8)) * $(4096)) + ((r110_z[:, None] % $(64)) * $(64)) + (r110_lane[None, :] % $(64))), dtype=tl.float32)
    r110_maximum = tl.max(r110_row_values, axis=1, keep_dims=true)
    r110_weights = tl.exp(r110_row_values - r110_maximum)
    r110_total = tl.sum(r110_weights, axis=1)
    r110_maximum = tl.max(r110_row_values, axis=1)
    r110_value = r110_maximum + tl.log(r110_total)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(512)])
    tl.store($(candidate_vlogsumexp_2_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(524288))
    r111_value = tl.where(((r111_z / $(4096)) % $(128)) < $(48), tl.load($(candidate_vmm_4_ptr) + (((((((r111_z / $(512)) % $(8)) * $(48)) + (((r111_z / $(4096)) % $(128)) - $(0))) % $(384)) * $(512)) + (r111_z % $(512))), dtype=tl.bfloat16), tl.where(((r111_z / $(4096)) % $(128)) < $(64), tl.load($(candidate_vmm_6_ptr) + (((((((r111_z / $(512)) % $(8)) * $(16)) + (((r111_z / $(4096)) % $(128)) - $(48))) % $(128)) * $(512)) + (r111_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_8_ptr) + (((((((r111_z / $(512)) % $(8)) * $(64)) + (((r111_z / $(4096)) % $(128)) - $(64))) % $(512)) * $(512)) + (r111_z % $(512))), dtype=tl.bfloat16)))
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(524288)])
    tl.store($(candidate_vcat_6_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(1024))
    r112_value = tl.where(((r112_z / $(8)) % $(128)) < $(48), tl.load($(candidate_vlogsumexp_ptr) + (((r112_z % $(8)) * $(48)) + ((((r112_z / $(8)) % $(128)) - $(0)) % $(48))), dtype=tl.float32), tl.where(((r112_z / $(8)) % $(128)) < $(64), tl.load($(candidate_vlogsumexp_1_ptr) + (((r112_z % $(8)) * $(16)) + ((((r112_z / $(8)) % $(128)) - $(48)) % $(16))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_2_ptr) + (((r112_z % $(8)) * $(64)) + ((((r112_z / $(8)) % $(128)) - $(64)) % $(64))), dtype=tl.float32)))
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(1024)])
    tl.store($(candidate_vcat_7_ptr) + r112_z, r112_value)
    r113_row = tl.arange($(128))
    r113_col = tl.arange($(128))
    r113_lane = tl.arange($(512))
    for r113_group in range($(8)) {
      r113_left = tl.load($(candidate_vcat_6_ptr) + (((((((r113_group % $(8)) * $(65536)) + (r113_row[:, None] * $(512)) + r113_lane[None, :]) / $(512)) % $(128)) * $(4096)) + ((((((r113_group % $(8)) * $(65536)) + (r113_row[:, None] * $(512)) + r113_lane[None, :]) / $(65536)) % $(8)) * $(512)) + ((((r113_group % $(8)) * $(65536)) + (r113_row[:, None] * $(512)) + r113_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r113_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r113_group % $(8)) * $(65536)) + (r113_lane[:, None] * $(128)) + r113_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r113_group % $(8)) * $(65536)) + (r113_lane[:, None] * $(128)) + r113_col[None, :]) % $(128)) * $(512)) + (((((r113_group % $(8)) * $(65536)) + (r113_lane[:, None] * $(128)) + r113_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r113_value = tl.dot(r113_left, r113_right)
      r113_value = r113_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_4_ptr) + ((r113_group * $(16384)) + (r113_row[:, None] * $(128)) + r113_col[None, :]), r113_value)
    }
    r114_row = tl.arange($(128))
    r114_col = tl.arange($(1024))
    r114_lane = tl.arange($(512))
    r114_left = tl.load($(input1_ptr) + ((((((r114_row[:, None] * $(512)) + r114_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r114_row[:, None] * $(512)) + r114_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r114_right = tl.load($(candidate_vto_copy_6_ptr) + (((r114_lane[:, None] % $(512)) * $(1024)) + (r114_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r114_value = tl.dot(r114_left, r114_right)
    r114_value = r114_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_10_ptr) + ((r114_row[:, None] * $(1024)) + r114_col[None, :]), r114_value)
    r115_row = tl.arange($(128))
    r115_col = tl.arange($(1024))
    r115_lane = tl.arange($(512))
    for r115_group in range($(1)) {
      r115_left = tl.load($(candidate_vto_copy_21_ptr) + ((((((r115_row[:, None] * $(512)) + r115_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r115_row[:, None] * $(512)) + r115_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r115_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r115_lane[:, None] * $(1024)) + r115_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r115_lane[:, None] * $(1024)) + r115_col[None, :]) % $(128)) * $(512)) + ((((r115_lane[:, None] * $(1024)) + r115_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r115_value = tl.dot(r115_left, r115_right)
      r115_value = r115_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r115_group * $(131072)) + (r115_row[:, None] * $(1024)) + r115_col[None, :]), r115_value)
    }
    r116_row = tl.arange($(128))
    r116_col = tl.arange($(1024))
    r116_lane = tl.arange($(512))
    for r116_group in range($(1)) {
      r116_left = tl.load($(candidate_vto_copy_21_ptr) + ((((((r116_row[:, None] * $(512)) + r116_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r116_row[:, None] * $(512)) + r116_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r116_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r116_lane[:, None] * $(1024)) + r116_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r116_lane[:, None] * $(1024)) + r116_col[None, :]) % $(128)) * $(512)) + ((((r116_lane[:, None] * $(1024)) + r116_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r116_value = tl.dot(r116_left, r116_right)
      r116_value = r116_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r116_group * $(131072)) + (r116_row[:, None] * $(1024)) + r116_col[None, :]), r116_value)
    }
    r117_z = tl.arange($(196608))
    r117_value = tl.where((r117_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r117_z / $(1536)) % $(128)) * $(1024)) + (((r117_z / $(192)) % $(8)) * $(128)) + ((r117_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r117_z / $(1536)) % $(128)) * $(1024)) + (((r117_z / $(192)) % $(8)) * $(128)) + ((r117_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r117_z / $(1536)) % $(128)) * $(1024)) + (((r117_z / $(192)) % $(8)) * $(128)) + ((r117_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_4_ptr) + ((((((((r117_z / $(1536)) % $(128)) * $(512)) + (((r117_z / $(192)) % $(8)) * $(64)) + ((r117_z % $(192)) - $(128))) / $(64)) % $(8)) * $(8192)) + (((((((r117_z / $(1536)) % $(128)) * $(512)) + (((r117_z / $(192)) % $(8)) * $(64)) + ((r117_z % $(192)) - $(128))) / $(512)) % $(128)) * $(64)) + (((((r117_z / $(1536)) % $(128)) * $(512)) + (((r117_z / $(192)) % $(8)) * $(64)) + ((r117_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r117_value = r117_value.to(tl.bfloat16)
    r117_value = tl.broadcast_to(r117_value, [$(196608)])
    tl.store($(candidate_vcat_8_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(196608))
    r118_value = tl.where((r118_z % $(192)) < $(128), tl.load($(candidate_vbmm_5_ptr) + ((((((((r118_z / $(1536)) % $(128)) * $(1024)) + (((r118_z / $(192)) % $(8)) * $(128)) + ((r118_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r118_z / $(1536)) % $(128)) * $(1024)) + (((r118_z / $(192)) % $(8)) * $(128)) + ((r118_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r118_z / $(1536)) % $(128)) * $(1024)) + (((r118_z / $(192)) % $(8)) * $(128)) + ((r118_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r118_z / $(1536)) % $(128)) * $(64)) + ((r118_z % $(192)) - $(128))) / $(64)) % $(128)) * $(64)) + (((((r118_z / $(1536)) % $(128)) * $(64)) + ((r118_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r118_value = r118_value.to(tl.bfloat16)
    r118_value = tl.broadcast_to(r118_value, [$(196608)])
    tl.store($(candidate_vcat_9_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(73728))
    r119_value = tl.load($(candidate_vcat_8_ptr) + ((((r119_z / $(192)) % $(48)) * $(1536)) + (((r119_z / $(9216)) % $(8)) * $(192)) + (r119_z % $(192))), dtype=tl.bfloat16)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(73728)])
    tl.store($(candidate_vto_copy_37_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(73728))
    r120_value = tl.load($(candidate_vcat_9_ptr) + ((((r120_z / $(192)) % $(48)) * $(1536)) + (((r120_z / $(9216)) % $(8)) * $(192)) + (r120_z % $(192))), dtype=tl.bfloat16)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(73728)])
    tl.store($(candidate_vto_copy_38_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(49152))
    r121_value = tl.load($(candidate_vbmm_6_ptr) + ((((r121_z / $(128)) % $(48)) * $(1024)) + (((r121_z / $(6144)) % $(8)) * $(128)) + (r121_z % $(128))), dtype=tl.bfloat16)
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(49152)])
    tl.store($(candidate_vto_copy_39_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(49152))
    r122_value = tl.load($(candidate_vmm_10_ptr) + ((((r122_z / $(128)) % $(48)) * $(1024)) + (((r122_z / $(6144)) % $(8)) * $(128)) + (r122_z % $(128))), dtype=tl.bfloat16)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(49152)])
    tl.store($(candidate_vto_copy_40_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(49152))
    r123_value = tl.load($(candidate_vbmm_4_ptr) + ((((r123_z / $(6144)) % $(8)) * $(16384)) + (((r123_z / $(128)) % $(48)) * $(128)) + (r123_z % $(128))), dtype=tl.bfloat16)
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(49152)])
    tl.store($(candidate_vto_copy_41_ptr) + r123_z, r123_value)
    r124_row = tl.arange($(48))
    r124_col = tl.arange($(48))
    r124_lane = tl.arange($(192))
    for r124_group in range($(8)) {
      r124_left = tl.load($(candidate_vto_copy_37_ptr) + (((((((r124_group % $(8)) * $(9216)) + (r124_row[:, None] * $(192)) + r124_lane[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r124_group % $(8)) * $(9216)) + (r124_row[:, None] * $(192)) + r124_lane[None, :]) / $(192)) % $(48)) * $(192)) + ((((r124_group % $(8)) * $(9216)) + (r124_row[:, None] * $(192)) + r124_lane[None, :]) % $(192))), dtype=tl.float32)
      r124_right = tl.load($(candidate_vto_copy_38_ptr) + (((((((r124_group % $(8)) * $(9216)) + (r124_lane[:, None] * $(48)) + r124_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r124_group % $(8)) * $(9216)) + (r124_lane[:, None] * $(48)) + r124_col[None, :]) % $(48)) * $(192)) + (((((r124_group % $(8)) * $(9216)) + (r124_lane[:, None] * $(48)) + r124_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r124_value = tl.dot(r124_left, r124_right)
      r124_value = r124_value.to(tl.float32)
      tl.store($(candidate_vbmm_7_ptr) + ((r124_group * $(2304)) + (r124_row[:, None] * $(48)) + r124_col[None, :]), r124_value)
    }
    r125_z = tl.arange($(18432))
    r125_value = (tl.load($(candidate_vbmm_7_ptr) + ((((r125_z / $(2304)) % $(8)) * $(2304)) + (((r125_z / $(48)) % $(48)) * $(48)) + (r125_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(18432)])
    tl.store($(candidate_vmul_24_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(48))
    r126_value = (tl.cast(r126_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r126_value = tl.broadcast_to(r126_value, [$(48)])
    tl.store($(candidate_vadd_12_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(48))
    r127_value = (tl.load($(candidate_vadd_12_ptr) + r127_z, dtype=tl.int64) - tl.full([], $(48), dtype=tl.int64))
    r127_value = tl.broadcast_to(r127_value, [$(48)])
    tl.store($(candidate_vsub_9_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(2304))
    r128_value = (tl.cast((r128_z % $(48)), tl.int64) > tl.load($(candidate_vsub_9_ptr) + ((r128_z / $(48)) % $(48)), dtype=tl.int64))
    r128_value = tl.broadcast_to(r128_value, [$(2304)])
    tl.store($(candidate_vgt_3_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(18432))
    r129_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r129_z / $(48)) % $(48)) * $(48)) + (r129_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r129_z / $(2304)) % $(8)) * $(2304)) + (((r129_z / $(48)) % $(48)) * $(48)) + (r129_z % $(48))), dtype=tl.float32))
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(18432))
    r130_value = (tl.load($(candidate_vmasked_fill_3_ptr) + ((((r130_z / $(2304)) % $(8)) * $(2304)) + (((r130_z / $(48)) % $(48)) * $(48)) + (r130_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_7_ptr) + ((((r130_z / $(48)) % $(48)) * $(8)) + ((r130_z / $(2304)) % $(8))), dtype=tl.float32))
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(18432)])
    tl.store($(candidate_vsub_10_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(18432))
    r131_value = tl.exp(tl.load($(candidate_vsub_10_ptr) + ((((r131_z / $(2304)) % $(8)) * $(2304)) + (((r131_z / $(48)) % $(48)) * $(48)) + (r131_z % $(48))), dtype=tl.float32))
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(18432)])
    tl.store($(candidate_vexp_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(49152))
    r132_value = (tl.load($(candidate_vto_copy_40_ptr) + ((((r132_z / $(6144)) % $(8)) * $(6144)) + (((r132_z / $(128)) % $(48)) * $(128)) + (r132_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_41_ptr) + ((((r132_z / $(6144)) % $(8)) * $(6144)) + (((r132_z / $(128)) % $(48)) * $(128)) + (r132_z % $(128))), dtype=tl.float32))
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(49152)])
    tl.store($(candidate_vmul_25_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(384))
    r133_lane = tl.arange($(128))
    r133_value = tl.sum(tl.load($(candidate_vmul_25_ptr) + ((((r133_z[:, None] / $(48)) % $(8)) * $(6144)) + ((r133_z[:, None] % $(48)) * $(128)) + r133_lane[None, :]), dtype=tl.float32), axis=1)
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(384)])
    tl.store($(candidate_vsum_1_ptr) + r133_z, r133_value)
    r134_row = tl.arange($(48))
    r134_col = tl.arange($(48))
    r134_lane = tl.arange($(128))
    for r134_group in range($(8)) {
      r134_left = tl.load($(candidate_vto_copy_40_ptr) + (((((((r134_group % $(8)) * $(6144)) + (r134_row[:, None] * $(128)) + r134_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r134_group % $(8)) * $(6144)) + (r134_row[:, None] * $(128)) + r134_lane[None, :]) / $(128)) % $(48)) * $(128)) + ((((r134_group % $(8)) * $(6144)) + (r134_row[:, None] * $(128)) + r134_lane[None, :]) % $(128))), dtype=tl.float32)
      r134_right = tl.load($(candidate_vto_copy_39_ptr) + (((((((r134_group % $(8)) * $(6144)) + (r134_lane[:, None] * $(48)) + r134_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r134_group % $(8)) * $(6144)) + (r134_lane[:, None] * $(48)) + r134_col[None, :]) % $(48)) * $(128)) + (((((r134_group % $(8)) * $(6144)) + (r134_lane[:, None] * $(48)) + r134_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r134_value = tl.dot(r134_left, r134_right)
      r134_value = r134_value.to(tl.float32)
      tl.store($(candidate_vbmm_8_ptr) + ((r134_group * $(2304)) + (r134_row[:, None] * $(48)) + r134_col[None, :]), r134_value)
    }
    r135_z = tl.arange($(18432))
    r135_value = (tl.load($(candidate_vbmm_8_ptr) + ((((r135_z / $(2304)) % $(8)) * $(2304)) + (((r135_z / $(48)) % $(48)) * $(48)) + (r135_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_1_ptr) + ((((r135_z / $(2304)) % $(8)) * $(48)) + ((r135_z / $(48)) % $(48))), dtype=tl.float32))
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(18432)])
    tl.store($(candidate_vsub_11_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(18432))
    r136_value = (tl.load($(candidate_vexp_ptr) + ((((r136_z / $(2304)) % $(8)) * $(2304)) + (((r136_z / $(48)) % $(48)) * $(48)) + (r136_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_11_ptr) + ((((r136_z / $(2304)) % $(8)) * $(2304)) + (((r136_z / $(48)) % $(48)) * $(48)) + (r136_z % $(48))), dtype=tl.float32))
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(18432)])
    tl.store($(candidate_vmul_26_ptr) + r136_z, r136_value)
    r137_row = tl.arange($(48))
    r137_col = tl.arange($(192))
    r137_lane = tl.arange($(48))
    for r137_group in range($(8)) {
      r137_left = tl.load($(candidate_vmul_26_ptr) + (((((((r137_group % $(8)) * $(2304)) + (r137_row[:, None] * $(48)) + r137_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r137_group % $(8)) * $(2304)) + (r137_row[:, None] * $(48)) + r137_lane[None, :]) / $(48)) % $(48)) * $(48)) + ((((r137_group % $(8)) * $(2304)) + (r137_row[:, None] * $(48)) + r137_lane[None, :]) % $(48))), dtype=tl.float32)
      r137_right = tl.load($(candidate_vto_copy_38_ptr) + (((((((r137_group % $(8)) * $(9216)) + (r137_lane[:, None] * $(192)) + r137_col[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r137_group % $(8)) * $(9216)) + (r137_lane[:, None] * $(192)) + r137_col[None, :]) / $(192)) % $(48)) * $(192)) + ((((r137_group % $(8)) * $(9216)) + (r137_lane[:, None] * $(192)) + r137_col[None, :]) % $(192))), dtype=tl.float32)
      r137_value = tl.dot(r137_left, r137_right)
      r137_value = r137_value.to(tl.float32)
      tl.store($(candidate_vbmm_9_ptr) + ((r137_group * $(9216)) + (r137_row[:, None] * $(192)) + r137_col[None, :]), r137_value)
    }
    r138_z = tl.arange($(73728))
    r138_value = (tl.load($(candidate_vbmm_9_ptr) + ((((r138_z / $(9216)) % $(8)) * $(9216)) + (((r138_z / $(192)) % $(48)) * $(192)) + (r138_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(73728)])
    tl.store($(candidate_vmul_27_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(73728))
    r139_value = tl.load($(candidate_vmul_27_ptr) + ((((r139_z / $(192)) % $(8)) * $(9216)) + (((r139_z / $(1536)) % $(48)) * $(192)) + (r139_z % $(192))), dtype=tl.float32)
    r139_value = r139_value.to(tl.bfloat16)
    r139_value = tl.broadcast_to(r139_value, [$(73728)])
    tl.store($(candidate_vto_copy_42_ptr) + r139_z, r139_value)
    r140_row = tl.arange($(48))
    r140_col = tl.arange($(192))
    r140_lane = tl.arange($(48))
    for r140_group in range($(8)) {
      r140_left = tl.load($(candidate_vmul_26_ptr) + (((((((r140_group % $(8)) * $(2304)) + (r140_row[:, None] * $(48)) + r140_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r140_group % $(8)) * $(2304)) + (r140_row[:, None] * $(48)) + r140_lane[None, :]) % $(48)) * $(48)) + (((((r140_group % $(8)) * $(2304)) + (r140_row[:, None] * $(48)) + r140_lane[None, :]) / $(48)) % $(48))), dtype=tl.float32)
      r140_right = tl.load($(candidate_vto_copy_37_ptr) + (((((((r140_group % $(8)) * $(9216)) + (r140_lane[:, None] * $(192)) + r140_col[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r140_group % $(8)) * $(9216)) + (r140_lane[:, None] * $(192)) + r140_col[None, :]) / $(192)) % $(48)) * $(192)) + ((((r140_group % $(8)) * $(9216)) + (r140_lane[:, None] * $(192)) + r140_col[None, :]) % $(192))), dtype=tl.float32)
      r140_value = tl.dot(r140_left, r140_right)
      r140_value = r140_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r140_group * $(9216)) + (r140_row[:, None] * $(192)) + r140_col[None, :]), r140_value)
    }
    r141_z = tl.arange($(73728))
    r141_value = (tl.load($(candidate_vbmm_10_ptr) + ((((r141_z / $(9216)) % $(8)) * $(9216)) + (((r141_z / $(192)) % $(48)) * $(192)) + (r141_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(73728)])
    tl.store($(candidate_vmul_28_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(73728))
    r142_value = tl.load($(candidate_vmul_28_ptr) + ((((r142_z / $(192)) % $(8)) * $(9216)) + (((r142_z / $(1536)) % $(48)) * $(192)) + (r142_z % $(192))), dtype=tl.float32)
    r142_value = r142_value.to(tl.bfloat16)
    r142_value = tl.broadcast_to(r142_value, [$(73728)])
    tl.store($(candidate_vto_copy_43_ptr) + r142_z, r142_value)
    r143_row = tl.arange($(48))
    r143_col = tl.arange($(128))
    r143_lane = tl.arange($(48))
    for r143_group in range($(8)) {
      r143_left = tl.load($(candidate_vexp_ptr) + (((((((r143_group % $(8)) * $(2304)) + (r143_row[:, None] * $(48)) + r143_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r143_group % $(8)) * $(2304)) + (r143_row[:, None] * $(48)) + r143_lane[None, :]) % $(48)) * $(48)) + (((((r143_group % $(8)) * $(2304)) + (r143_row[:, None] * $(48)) + r143_lane[None, :]) / $(48)) % $(48))), dtype=tl.float32)
      r143_right = tl.load($(candidate_vto_copy_40_ptr) + (((((((r143_group % $(8)) * $(6144)) + (r143_lane[:, None] * $(128)) + r143_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r143_group % $(8)) * $(6144)) + (r143_lane[:, None] * $(128)) + r143_col[None, :]) / $(128)) % $(48)) * $(128)) + ((((r143_group % $(8)) * $(6144)) + (r143_lane[:, None] * $(128)) + r143_col[None, :]) % $(128))), dtype=tl.float32)
      r143_value = tl.dot(r143_left, r143_right)
      r143_value = r143_value.to(tl.float32)
      tl.store($(candidate_vbmm_11_ptr) + ((r143_group * $(6144)) + (r143_row[:, None] * $(128)) + r143_col[None, :]), r143_value)
    }
    r144_z = tl.arange($(49152))
    r144_value = tl.load($(candidate_vbmm_11_ptr) + ((((r144_z / $(128)) % $(8)) * $(6144)) + (((r144_z / $(1024)) % $(48)) * $(128)) + (r144_z % $(128))), dtype=tl.float32)
    r144_value = r144_value.to(tl.bfloat16)
    r144_value = tl.broadcast_to(r144_value, [$(49152)])
    tl.store($(candidate_vto_copy_44_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(24576))
    r145_value = tl.load($(candidate_vcat_8_ptr) + ((((r145_z / $(192)) % $(16)) * $(1536)) + (((r145_z / $(3072)) % $(8)) * $(192)) + (r145_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(24576)])
    tl.store($(candidate_vto_copy_45_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(24576))
    r146_value = tl.load($(candidate_vcat_9_ptr) + ((((r146_z / $(192)) % $(16)) * $(1536)) + (((r146_z / $(3072)) % $(8)) * $(192)) + (r146_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r146_value = r146_value.to(tl.float32)
    r146_value = tl.broadcast_to(r146_value, [$(24576)])
    tl.store($(candidate_vto_copy_46_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(16384))
    r147_value = tl.load($(candidate_vbmm_6_ptr) + ((((r147_z / $(128)) % $(16)) * $(1024)) + (((r147_z / $(2048)) % $(8)) * $(128)) + (r147_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r147_value = r147_value.to(tl.float32)
    r147_value = tl.broadcast_to(r147_value, [$(16384)])
    tl.store($(candidate_vto_copy_47_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(16384))
    r148_value = tl.load($(candidate_vmm_10_ptr) + ((((r148_z / $(128)) % $(16)) * $(1024)) + (((r148_z / $(2048)) % $(8)) * $(128)) + (r148_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(16384)])
    tl.store($(candidate_vto_copy_48_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(16384))
    r149_value = tl.load($(candidate_vbmm_4_ptr) + ((((r149_z / $(2048)) % $(8)) * $(16384)) + (((r149_z / $(128)) % $(16)) * $(128)) + (r149_z % $(128)) + $(6144)), dtype=tl.bfloat16)
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(16384)])
    tl.store($(candidate_vto_copy_49_ptr) + r149_z, r149_value)
    r150_row = tl.arange($(16))
    r150_col = tl.arange($(16))
    r150_lane = tl.arange($(192))
    for r150_group in range($(8)) {
      r150_left = tl.load($(candidate_vto_copy_45_ptr) + (((((((r150_group % $(8)) * $(3072)) + (r150_row[:, None] * $(192)) + r150_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r150_group % $(8)) * $(3072)) + (r150_row[:, None] * $(192)) + r150_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r150_group % $(8)) * $(3072)) + (r150_row[:, None] * $(192)) + r150_lane[None, :]) % $(192))), dtype=tl.float32)
      r150_right = tl.load($(candidate_vto_copy_46_ptr) + (((((((r150_group % $(8)) * $(3072)) + (r150_lane[:, None] * $(16)) + r150_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r150_group % $(8)) * $(3072)) + (r150_lane[:, None] * $(16)) + r150_col[None, :]) % $(16)) * $(192)) + (((((r150_group % $(8)) * $(3072)) + (r150_lane[:, None] * $(16)) + r150_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r150_value = tl.dot(r150_left, r150_right)
      r150_value = r150_value.to(tl.float32)
      tl.store($(candidate_vbmm_12_ptr) + ((r150_group * $(256)) + (r150_row[:, None] * $(16)) + r150_col[None, :]), r150_value)
    }
    r151_z = tl.arange($(2048))
    r151_value = (tl.load($(candidate_vbmm_12_ptr) + ((((r151_z / $(256)) % $(8)) * $(256)) + (((r151_z / $(16)) % $(16)) * $(16)) + (r151_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(2048)])
    tl.store($(candidate_vmul_29_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(16))
    r152_value = (tl.cast(r152_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r152_value = tl.broadcast_to(r152_value, [$(16)])
    tl.store($(candidate_vadd_13_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(16))
    r153_value = (tl.load($(candidate_vadd_13_ptr) + r153_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r153_value = tl.broadcast_to(r153_value, [$(16)])
    tl.store($(candidate_vsub_12_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(256))
    r154_value = (tl.cast((r154_z % $(16)), tl.int64) > tl.load($(candidate_vsub_12_ptr) + ((r154_z / $(16)) % $(16)), dtype=tl.int64))
    r154_value = tl.broadcast_to(r154_value, [$(256)])
    tl.store($(candidate_vgt_4_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(2048))
    r155_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r155_z / $(16)) % $(16)) * $(16)) + (r155_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_29_ptr) + ((((r155_z / $(256)) % $(8)) * $(256)) + (((r155_z / $(16)) % $(16)) * $(16)) + (r155_z % $(16))), dtype=tl.float32))
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(2048))
    r156_value = (tl.load($(candidate_vmasked_fill_4_ptr) + ((((r156_z / $(256)) % $(8)) * $(256)) + (((r156_z / $(16)) % $(16)) * $(16)) + (r156_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_7_ptr) + ((((r156_z / $(16)) % $(16)) * $(8)) + ((r156_z / $(256)) % $(8)) + $(384)), dtype=tl.float32))
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(2048)])
    tl.store($(candidate_vsub_13_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(2048))
    r157_value = tl.exp(tl.load($(candidate_vsub_13_ptr) + ((((r157_z / $(256)) % $(8)) * $(256)) + (((r157_z / $(16)) % $(16)) * $(16)) + (r157_z % $(16))), dtype=tl.float32))
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(2048)])
    tl.store($(candidate_vexp_1_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(16384))
    r158_value = (tl.load($(candidate_vto_copy_48_ptr) + ((((r158_z / $(2048)) % $(8)) * $(2048)) + (((r158_z / $(128)) % $(16)) * $(128)) + (r158_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_49_ptr) + ((((r158_z / $(2048)) % $(8)) * $(2048)) + (((r158_z / $(128)) % $(16)) * $(128)) + (r158_z % $(128))), dtype=tl.float32))
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(16384)])
    tl.store($(candidate_vmul_30_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(128))
    r159_lane = tl.arange($(128))
    r159_value = tl.sum(tl.load($(candidate_vmul_30_ptr) + ((((r159_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r159_z[:, None] % $(16)) * $(128)) + r159_lane[None, :]), dtype=tl.float32), axis=1)
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(128)])
    tl.store($(candidate_vsum_2_ptr) + r159_z, r159_value)
    r160_row = tl.arange($(16))
    r160_col = tl.arange($(16))
    r160_lane = tl.arange($(128))
    for r160_group in range($(8)) {
      r160_left = tl.load($(candidate_vto_copy_48_ptr) + (((((((r160_group % $(8)) * $(2048)) + (r160_row[:, None] * $(128)) + r160_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r160_group % $(8)) * $(2048)) + (r160_row[:, None] * $(128)) + r160_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r160_group % $(8)) * $(2048)) + (r160_row[:, None] * $(128)) + r160_lane[None, :]) % $(128))), dtype=tl.float32)
      r160_right = tl.load($(candidate_vto_copy_47_ptr) + (((((((r160_group % $(8)) * $(2048)) + (r160_lane[:, None] * $(16)) + r160_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r160_group % $(8)) * $(2048)) + (r160_lane[:, None] * $(16)) + r160_col[None, :]) % $(16)) * $(128)) + (((((r160_group % $(8)) * $(2048)) + (r160_lane[:, None] * $(16)) + r160_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r160_value = tl.dot(r160_left, r160_right)
      r160_value = r160_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r160_group * $(256)) + (r160_row[:, None] * $(16)) + r160_col[None, :]), r160_value)
    }
    r161_z = tl.arange($(2048))
    r161_value = (tl.load($(candidate_vbmm_13_ptr) + ((((r161_z / $(256)) % $(8)) * $(256)) + (((r161_z / $(16)) % $(16)) * $(16)) + (r161_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_2_ptr) + ((((r161_z / $(256)) % $(8)) * $(16)) + ((r161_z / $(16)) % $(16))), dtype=tl.float32))
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(2048)])
    tl.store($(candidate_vsub_14_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(2048))
    r162_value = (tl.load($(candidate_vexp_1_ptr) + ((((r162_z / $(256)) % $(8)) * $(256)) + (((r162_z / $(16)) % $(16)) * $(16)) + (r162_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_14_ptr) + ((((r162_z / $(256)) % $(8)) * $(256)) + (((r162_z / $(16)) % $(16)) * $(16)) + (r162_z % $(16))), dtype=tl.float32))
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(2048)])
    tl.store($(candidate_vmul_31_ptr) + r162_z, r162_value)
    r163_row = tl.arange($(16))
    r163_col = tl.arange($(192))
    r163_lane = tl.arange($(16))
    for r163_group in range($(8)) {
      r163_left = tl.load($(candidate_vmul_31_ptr) + (((((((r163_group % $(8)) * $(256)) + (r163_row[:, None] * $(16)) + r163_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r163_group % $(8)) * $(256)) + (r163_row[:, None] * $(16)) + r163_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r163_group % $(8)) * $(256)) + (r163_row[:, None] * $(16)) + r163_lane[None, :]) % $(16))), dtype=tl.float32)
      r163_right = tl.load($(candidate_vto_copy_46_ptr) + (((((((r163_group % $(8)) * $(3072)) + (r163_lane[:, None] * $(192)) + r163_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r163_group % $(8)) * $(3072)) + (r163_lane[:, None] * $(192)) + r163_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r163_group % $(8)) * $(3072)) + (r163_lane[:, None] * $(192)) + r163_col[None, :]) % $(192))), dtype=tl.float32)
      r163_value = tl.dot(r163_left, r163_right)
      r163_value = r163_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r163_group * $(3072)) + (r163_row[:, None] * $(192)) + r163_col[None, :]), r163_value)
    }
    r164_z = tl.arange($(24576))
    r164_value = (tl.load($(candidate_vbmm_14_ptr) + ((((r164_z / $(3072)) % $(8)) * $(3072)) + (((r164_z / $(192)) % $(16)) * $(192)) + (r164_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(24576)])
    tl.store($(candidate_vmul_32_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(24576))
    r165_value = tl.load($(candidate_vmul_32_ptr) + ((((r165_z / $(192)) % $(8)) * $(3072)) + (((r165_z / $(1536)) % $(16)) * $(192)) + (r165_z % $(192))), dtype=tl.float32)
    r165_value = r165_value.to(tl.bfloat16)
    r165_value = tl.broadcast_to(r165_value, [$(24576)])
    tl.store($(candidate_vto_copy_50_ptr) + r165_z, r165_value)
    r166_row = tl.arange($(16))
    r166_col = tl.arange($(192))
    r166_lane = tl.arange($(16))
    for r166_group in range($(8)) {
      r166_left = tl.load($(candidate_vmul_31_ptr) + (((((((r166_group % $(8)) * $(256)) + (r166_row[:, None] * $(16)) + r166_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r166_group % $(8)) * $(256)) + (r166_row[:, None] * $(16)) + r166_lane[None, :]) % $(16)) * $(16)) + (((((r166_group % $(8)) * $(256)) + (r166_row[:, None] * $(16)) + r166_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r166_right = tl.load($(candidate_vto_copy_45_ptr) + (((((((r166_group % $(8)) * $(3072)) + (r166_lane[:, None] * $(192)) + r166_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r166_group % $(8)) * $(3072)) + (r166_lane[:, None] * $(192)) + r166_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r166_group % $(8)) * $(3072)) + (r166_lane[:, None] * $(192)) + r166_col[None, :]) % $(192))), dtype=tl.float32)
      r166_value = tl.dot(r166_left, r166_right)
      r166_value = r166_value.to(tl.float32)
      tl.store($(candidate_vbmm_15_ptr) + ((r166_group * $(3072)) + (r166_row[:, None] * $(192)) + r166_col[None, :]), r166_value)
    }
    r167_z = tl.arange($(24576))
    r167_value = (tl.load($(candidate_vbmm_15_ptr) + ((((r167_z / $(3072)) % $(8)) * $(3072)) + (((r167_z / $(192)) % $(16)) * $(192)) + (r167_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(24576)])
    tl.store($(candidate_vmul_33_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(24576))
    r168_value = tl.load($(candidate_vmul_33_ptr) + ((((r168_z / $(192)) % $(8)) * $(3072)) + (((r168_z / $(1536)) % $(16)) * $(192)) + (r168_z % $(192))), dtype=tl.float32)
    r168_value = r168_value.to(tl.bfloat16)
    r168_value = tl.broadcast_to(r168_value, [$(24576)])
    tl.store($(candidate_vto_copy_51_ptr) + r168_z, r168_value)
    r169_row = tl.arange($(16))
    r169_col = tl.arange($(128))
    r169_lane = tl.arange($(16))
    for r169_group in range($(8)) {
      r169_left = tl.load($(candidate_vexp_1_ptr) + (((((((r169_group % $(8)) * $(256)) + (r169_row[:, None] * $(16)) + r169_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r169_group % $(8)) * $(256)) + (r169_row[:, None] * $(16)) + r169_lane[None, :]) % $(16)) * $(16)) + (((((r169_group % $(8)) * $(256)) + (r169_row[:, None] * $(16)) + r169_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r169_right = tl.load($(candidate_vto_copy_48_ptr) + (((((((r169_group % $(8)) * $(2048)) + (r169_lane[:, None] * $(128)) + r169_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r169_group % $(8)) * $(2048)) + (r169_lane[:, None] * $(128)) + r169_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r169_group % $(8)) * $(2048)) + (r169_lane[:, None] * $(128)) + r169_col[None, :]) % $(128))), dtype=tl.float32)
      r169_value = tl.dot(r169_left, r169_right)
      r169_value = r169_value.to(tl.float32)
      tl.store($(candidate_vbmm_16_ptr) + ((r169_group * $(2048)) + (r169_row[:, None] * $(128)) + r169_col[None, :]), r169_value)
    }
    r170_z = tl.arange($(16384))
    r170_value = tl.load($(candidate_vbmm_16_ptr) + ((((r170_z / $(128)) % $(8)) * $(2048)) + (((r170_z / $(1024)) % $(16)) * $(128)) + (r170_z % $(128))), dtype=tl.float32)
    r170_value = r170_value.to(tl.bfloat16)
    r170_value = tl.broadcast_to(r170_value, [$(16384)])
    tl.store($(candidate_vto_copy_52_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(98304))
    r171_value = tl.load($(candidate_vcat_8_ptr) + ((((r171_z / $(192)) % $(64)) * $(1536)) + (((r171_z / $(12288)) % $(8)) * $(192)) + (r171_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r171_value = r171_value.to(tl.float32)
    r171_value = tl.broadcast_to(r171_value, [$(98304)])
    tl.store($(candidate_vto_copy_53_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(98304))
    r172_value = tl.load($(candidate_vcat_9_ptr) + ((((r172_z / $(192)) % $(64)) * $(1536)) + (((r172_z / $(12288)) % $(8)) * $(192)) + (r172_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(98304)])
    tl.store($(candidate_vto_copy_54_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(65536))
    r173_value = tl.load($(candidate_vbmm_6_ptr) + ((((r173_z / $(128)) % $(64)) * $(1024)) + (((r173_z / $(8192)) % $(8)) * $(128)) + (r173_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r173_value = r173_value.to(tl.float32)
    r173_value = tl.broadcast_to(r173_value, [$(65536)])
    tl.store($(candidate_vto_copy_55_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(65536))
    r174_value = tl.load($(candidate_vmm_10_ptr) + ((((r174_z / $(128)) % $(64)) * $(1024)) + (((r174_z / $(8192)) % $(8)) * $(128)) + (r174_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r174_value = r174_value.to(tl.float32)
    r174_value = tl.broadcast_to(r174_value, [$(65536)])
    tl.store($(candidate_vto_copy_56_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(65536))
    r175_value = tl.load($(candidate_vbmm_4_ptr) + ((((r175_z / $(8192)) % $(8)) * $(16384)) + (((r175_z / $(128)) % $(64)) * $(128)) + (r175_z % $(128)) + $(8192)), dtype=tl.bfloat16)
    r175_value = r175_value.to(tl.float32)
    r175_value = tl.broadcast_to(r175_value, [$(65536)])
    tl.store($(candidate_vto_copy_57_ptr) + r175_z, r175_value)
    r176_row = tl.arange($(64))
    r176_col = tl.arange($(64))
    r176_lane = tl.arange($(192))
    for r176_group in range($(8)) {
      r176_left = tl.load($(candidate_vto_copy_53_ptr) + (((((((r176_group % $(8)) * $(12288)) + (r176_row[:, None] * $(192)) + r176_lane[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r176_group % $(8)) * $(12288)) + (r176_row[:, None] * $(192)) + r176_lane[None, :]) / $(192)) % $(64)) * $(192)) + ((((r176_group % $(8)) * $(12288)) + (r176_row[:, None] * $(192)) + r176_lane[None, :]) % $(192))), dtype=tl.float32)
      r176_right = tl.load($(candidate_vto_copy_54_ptr) + (((((((r176_group % $(8)) * $(12288)) + (r176_lane[:, None] * $(64)) + r176_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r176_group % $(8)) * $(12288)) + (r176_lane[:, None] * $(64)) + r176_col[None, :]) % $(64)) * $(192)) + (((((r176_group % $(8)) * $(12288)) + (r176_lane[:, None] * $(64)) + r176_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r176_value = tl.dot(r176_left, r176_right)
      r176_value = r176_value.to(tl.float32)
      tl.store($(candidate_vbmm_17_ptr) + ((r176_group * $(4096)) + (r176_row[:, None] * $(64)) + r176_col[None, :]), r176_value)
    }
    r177_z = tl.arange($(32768))
    r177_value = (tl.load($(candidate_vbmm_17_ptr) + ((((r177_z / $(4096)) % $(8)) * $(4096)) + (((r177_z / $(64)) % $(64)) * $(64)) + (r177_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(32768)])
    tl.store($(candidate_vmul_34_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(64))
    r178_value = (tl.cast(r178_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r178_value = tl.broadcast_to(r178_value, [$(64)])
    tl.store($(candidate_vadd_14_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(64))
    r179_value = (tl.load($(candidate_vadd_14_ptr) + r179_z, dtype=tl.int64) - tl.full([], $(64), dtype=tl.int64))
    r179_value = tl.broadcast_to(r179_value, [$(64)])
    tl.store($(candidate_vsub_15_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(4096))
    r180_value = (tl.cast((r180_z % $(64)), tl.int64) > tl.load($(candidate_vsub_15_ptr) + ((r180_z / $(64)) % $(64)), dtype=tl.int64))
    r180_value = tl.broadcast_to(r180_value, [$(4096)])
    tl.store($(candidate_vgt_5_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(32768))
    r181_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r181_z / $(64)) % $(64)) * $(64)) + (r181_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_34_ptr) + ((((r181_z / $(4096)) % $(8)) * $(4096)) + (((r181_z / $(64)) % $(64)) * $(64)) + (r181_z % $(64))), dtype=tl.float32))
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(32768))
    r182_value = (tl.load($(candidate_vmasked_fill_5_ptr) + ((((r182_z / $(4096)) % $(8)) * $(4096)) + (((r182_z / $(64)) % $(64)) * $(64)) + (r182_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_7_ptr) + ((((r182_z / $(64)) % $(64)) * $(8)) + ((r182_z / $(4096)) % $(8)) + $(512)), dtype=tl.float32))
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(32768)])
    tl.store($(candidate_vsub_16_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(32768))
    r183_value = tl.exp(tl.load($(candidate_vsub_16_ptr) + ((((r183_z / $(4096)) % $(8)) * $(4096)) + (((r183_z / $(64)) % $(64)) * $(64)) + (r183_z % $(64))), dtype=tl.float32))
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(32768)])
    tl.store($(candidate_vexp_2_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(65536))
    r184_value = (tl.load($(candidate_vto_copy_56_ptr) + ((((r184_z / $(8192)) % $(8)) * $(8192)) + (((r184_z / $(128)) % $(64)) * $(128)) + (r184_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_57_ptr) + ((((r184_z / $(8192)) % $(8)) * $(8192)) + (((r184_z / $(128)) % $(64)) * $(128)) + (r184_z % $(128))), dtype=tl.float32))
    r184_value = r184_value.to(tl.float32)
    r184_value = tl.broadcast_to(r184_value, [$(65536)])
    tl.store($(candidate_vmul_35_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(512))
    r185_lane = tl.arange($(128))
    r185_value = tl.sum(tl.load($(candidate_vmul_35_ptr) + ((((r185_z[:, None] / $(64)) % $(8)) * $(8192)) + ((r185_z[:, None] % $(64)) * $(128)) + r185_lane[None, :]), dtype=tl.float32), axis=1)
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(512)])
    tl.store($(candidate_vsum_3_ptr) + r185_z, r185_value)
    r186_row = tl.arange($(64))
    r186_col = tl.arange($(64))
    r186_lane = tl.arange($(128))
    for r186_group in range($(8)) {
      r186_left = tl.load($(candidate_vto_copy_56_ptr) + (((((((r186_group % $(8)) * $(8192)) + (r186_row[:, None] * $(128)) + r186_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r186_group % $(8)) * $(8192)) + (r186_row[:, None] * $(128)) + r186_lane[None, :]) / $(128)) % $(64)) * $(128)) + ((((r186_group % $(8)) * $(8192)) + (r186_row[:, None] * $(128)) + r186_lane[None, :]) % $(128))), dtype=tl.float32)
      r186_right = tl.load($(candidate_vto_copy_55_ptr) + (((((((r186_group % $(8)) * $(8192)) + (r186_lane[:, None] * $(64)) + r186_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r186_group % $(8)) * $(8192)) + (r186_lane[:, None] * $(64)) + r186_col[None, :]) % $(64)) * $(128)) + (((((r186_group % $(8)) * $(8192)) + (r186_lane[:, None] * $(64)) + r186_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r186_value = tl.dot(r186_left, r186_right)
      r186_value = r186_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r186_group * $(4096)) + (r186_row[:, None] * $(64)) + r186_col[None, :]), r186_value)
    }
    r187_z = tl.arange($(32768))
    r187_value = (tl.load($(candidate_vbmm_18_ptr) + ((((r187_z / $(4096)) % $(8)) * $(4096)) + (((r187_z / $(64)) % $(64)) * $(64)) + (r187_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_3_ptr) + ((((r187_z / $(4096)) % $(8)) * $(64)) + ((r187_z / $(64)) % $(64))), dtype=tl.float32))
    r187_value = r187_value.to(tl.float32)
    r187_value = tl.broadcast_to(r187_value, [$(32768)])
    tl.store($(candidate_vsub_17_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(32768))
    r188_value = (tl.load($(candidate_vexp_2_ptr) + ((((r188_z / $(4096)) % $(8)) * $(4096)) + (((r188_z / $(64)) % $(64)) * $(64)) + (r188_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_17_ptr) + ((((r188_z / $(4096)) % $(8)) * $(4096)) + (((r188_z / $(64)) % $(64)) * $(64)) + (r188_z % $(64))), dtype=tl.float32))
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(32768)])
    tl.store($(candidate_vmul_36_ptr) + r188_z, r188_value)
    r189_row = tl.arange($(64))
    r189_col = tl.arange($(192))
    r189_lane = tl.arange($(64))
    for r189_group in range($(8)) {
      r189_left = tl.load($(candidate_vmul_36_ptr) + (((((((r189_group % $(8)) * $(4096)) + (r189_row[:, None] * $(64)) + r189_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r189_group % $(8)) * $(4096)) + (r189_row[:, None] * $(64)) + r189_lane[None, :]) / $(64)) % $(64)) * $(64)) + ((((r189_group % $(8)) * $(4096)) + (r189_row[:, None] * $(64)) + r189_lane[None, :]) % $(64))), dtype=tl.float32)
      r189_right = tl.load($(candidate_vto_copy_54_ptr) + (((((((r189_group % $(8)) * $(12288)) + (r189_lane[:, None] * $(192)) + r189_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r189_group % $(8)) * $(12288)) + (r189_lane[:, None] * $(192)) + r189_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r189_group % $(8)) * $(12288)) + (r189_lane[:, None] * $(192)) + r189_col[None, :]) % $(192))), dtype=tl.float32)
      r189_value = tl.dot(r189_left, r189_right)
      r189_value = r189_value.to(tl.float32)
      tl.store($(candidate_vbmm_19_ptr) + ((r189_group * $(12288)) + (r189_row[:, None] * $(192)) + r189_col[None, :]), r189_value)
    }
    r190_z = tl.arange($(98304))
    r190_value = (tl.load($(candidate_vbmm_19_ptr) + ((((r190_z / $(12288)) % $(8)) * $(12288)) + (((r190_z / $(192)) % $(64)) * $(192)) + (r190_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r190_value = r190_value.to(tl.float32)
    r190_value = tl.broadcast_to(r190_value, [$(98304)])
    tl.store($(candidate_vmul_37_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(98304))
    r191_value = tl.load($(candidate_vmul_37_ptr) + ((((r191_z / $(192)) % $(8)) * $(12288)) + (((r191_z / $(1536)) % $(64)) * $(192)) + (r191_z % $(192))), dtype=tl.float32)
    r191_value = r191_value.to(tl.bfloat16)
    r191_value = tl.broadcast_to(r191_value, [$(98304)])
    tl.store($(candidate_vto_copy_58_ptr) + r191_z, r191_value)
    r192_row = tl.arange($(64))
    r192_col = tl.arange($(192))
    r192_lane = tl.arange($(64))
    for r192_group in range($(8)) {
      r192_left = tl.load($(candidate_vmul_36_ptr) + (((((((r192_group % $(8)) * $(4096)) + (r192_row[:, None] * $(64)) + r192_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r192_group % $(8)) * $(4096)) + (r192_row[:, None] * $(64)) + r192_lane[None, :]) % $(64)) * $(64)) + (((((r192_group % $(8)) * $(4096)) + (r192_row[:, None] * $(64)) + r192_lane[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r192_right = tl.load($(candidate_vto_copy_53_ptr) + (((((((r192_group % $(8)) * $(12288)) + (r192_lane[:, None] * $(192)) + r192_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r192_group % $(8)) * $(12288)) + (r192_lane[:, None] * $(192)) + r192_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r192_group % $(8)) * $(12288)) + (r192_lane[:, None] * $(192)) + r192_col[None, :]) % $(192))), dtype=tl.float32)
      r192_value = tl.dot(r192_left, r192_right)
      r192_value = r192_value.to(tl.float32)
      tl.store($(candidate_vbmm_20_ptr) + ((r192_group * $(12288)) + (r192_row[:, None] * $(192)) + r192_col[None, :]), r192_value)
    }
    r193_z = tl.arange($(98304))
    r193_value = (tl.load($(candidate_vbmm_20_ptr) + ((((r193_z / $(12288)) % $(8)) * $(12288)) + (((r193_z / $(192)) % $(64)) * $(192)) + (r193_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r193_value = r193_value.to(tl.float32)
    r193_value = tl.broadcast_to(r193_value, [$(98304)])
    tl.store($(candidate_vmul_38_ptr) + r193_z, r193_value)
    r194_z = tl.arange($(98304))
    r194_value = tl.load($(candidate_vmul_38_ptr) + ((((r194_z / $(192)) % $(8)) * $(12288)) + (((r194_z / $(1536)) % $(64)) * $(192)) + (r194_z % $(192))), dtype=tl.float32)
    r194_value = r194_value.to(tl.bfloat16)
    r194_value = tl.broadcast_to(r194_value, [$(98304)])
    tl.store($(candidate_vto_copy_59_ptr) + r194_z, r194_value)
    r195_row = tl.arange($(64))
    r195_col = tl.arange($(128))
    r195_lane = tl.arange($(64))
    for r195_group in range($(8)) {
      r195_left = tl.load($(candidate_vexp_2_ptr) + (((((((r195_group % $(8)) * $(4096)) + (r195_row[:, None] * $(64)) + r195_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r195_group % $(8)) * $(4096)) + (r195_row[:, None] * $(64)) + r195_lane[None, :]) % $(64)) * $(64)) + (((((r195_group % $(8)) * $(4096)) + (r195_row[:, None] * $(64)) + r195_lane[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r195_right = tl.load($(candidate_vto_copy_56_ptr) + (((((((r195_group % $(8)) * $(8192)) + (r195_lane[:, None] * $(128)) + r195_col[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r195_group % $(8)) * $(8192)) + (r195_lane[:, None] * $(128)) + r195_col[None, :]) / $(128)) % $(64)) * $(128)) + ((((r195_group % $(8)) * $(8192)) + (r195_lane[:, None] * $(128)) + r195_col[None, :]) % $(128))), dtype=tl.float32)
      r195_value = tl.dot(r195_left, r195_right)
      r195_value = r195_value.to(tl.float32)
      tl.store($(candidate_vbmm_21_ptr) + ((r195_group * $(8192)) + (r195_row[:, None] * $(128)) + r195_col[None, :]), r195_value)
    }
    r196_z = tl.arange($(65536))
    r196_value = tl.load($(candidate_vbmm_21_ptr) + ((((r196_z / $(128)) % $(8)) * $(8192)) + (((r196_z / $(1024)) % $(64)) * $(128)) + (r196_z % $(128))), dtype=tl.float32)
    r196_value = r196_value.to(tl.bfloat16)
    r196_value = tl.broadcast_to(r196_value, [$(65536)])
    tl.store($(candidate_vto_copy_60_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(196608))
    r197_value = tl.where(((r197_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vto_copy_42_ptr) + ((((((r197_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r197_z / $(192)) % $(8)) * $(192)) + (r197_z % $(192))), dtype=tl.bfloat16), tl.where(((r197_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vto_copy_50_ptr) + ((((((r197_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r197_z / $(192)) % $(8)) * $(192)) + (r197_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_58_ptr) + ((((((r197_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r197_z / $(192)) % $(8)) * $(192)) + (r197_z % $(192))), dtype=tl.bfloat16)))
    r197_value = r197_value.to(tl.bfloat16)
    r197_value = tl.broadcast_to(r197_value, [$(196608)])
    tl.store($(candidate_vcat_10_ptr) + r197_z, r197_value)
    r198_z = tl.arange($(196608))
    r198_value = tl.where(((r198_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vto_copy_43_ptr) + ((((((r198_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r198_z / $(192)) % $(8)) * $(192)) + (r198_z % $(192))), dtype=tl.bfloat16), tl.where(((r198_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vto_copy_51_ptr) + ((((((r198_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r198_z / $(192)) % $(8)) * $(192)) + (r198_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_59_ptr) + ((((((r198_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r198_z / $(192)) % $(8)) * $(192)) + (r198_z % $(192))), dtype=tl.bfloat16)))
    r198_value = r198_value.to(tl.bfloat16)
    r198_value = tl.broadcast_to(r198_value, [$(196608)])
    tl.store($(candidate_vcat_11_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(131072))
    r199_value = tl.where(((r199_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vto_copy_44_ptr) + ((((((r199_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r199_z / $(128)) % $(8)) * $(128)) + (r199_z % $(128))), dtype=tl.bfloat16), tl.where(((r199_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vto_copy_52_ptr) + ((((((r199_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r199_z / $(128)) % $(8)) * $(128)) + (r199_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_60_ptr) + ((((((r199_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r199_z / $(128)) % $(8)) * $(128)) + (r199_z % $(128))), dtype=tl.bfloat16)))
    r199_value = r199_value.to(tl.bfloat16)
    r199_value = tl.broadcast_to(r199_value, [$(131072)])
    tl.store($(candidate_vcat_12_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(8192))
    r200_lane = tl.arange($(8))
    r200_value = tl.sum(tl.load($(candidate_vcat_11_ptr) + ((((r200_z[:, None] / $(64)) % $(128)) * $(1536)) + (r200_lane[None, :] * $(192)) + (r200_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r200_value = r200_value.to(tl.bfloat16)
    r200_value = tl.broadcast_to(r200_value, [$(8192)])
    tl.store($(candidate_vsum_4_ptr) + r200_z, r200_value)
    r201_row = tl.arange($(128))
    r201_col = tl.arange($(512))
    r201_lane = tl.arange($(1024))
    for r201_group in range($(1)) {
      r201_left = tl.load($(candidate_vcat_11_ptr) + ((((((r201_row[:, None] * $(1024)) + r201_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r201_row[:, None] * $(1024)) + r201_lane[None, :]) % $(8)) * $(192)) + ((((r201_row[:, None] * $(1024)) + r201_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r201_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r201_lane[:, None] * $(512)) + r201_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r201_lane[:, None] * $(512)) + r201_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r201_lane[:, None] * $(512)) + r201_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r201_value = tl.dot(r201_left, r201_right)
      r201_value = r201_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_22_ptr) + ((r201_group * $(65536)) + (r201_row[:, None] * $(512)) + r201_col[None, :]), r201_value)
    }
    r202_row = tl.arange($(128))
    r202_col = tl.arange($(512))
    r202_lane = tl.arange($(1024))
    for r202_group in range($(1)) {
      r202_left = tl.load($(candidate_vcat_12_ptr) + ((((((r202_row[:, None] * $(1024)) + r202_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r202_row[:, None] * $(1024)) + r202_lane[None, :]) % $(8)) * $(128)) + ((((r202_row[:, None] * $(1024)) + r202_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r202_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r202_lane[:, None] * $(512)) + r202_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r202_lane[:, None] * $(512)) + r202_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r202_lane[:, None] * $(512)) + r202_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r202_value = tl.dot(r202_left, r202_right)
      r202_value = r202_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_23_ptr) + ((r202_group * $(65536)) + (r202_row[:, None] * $(512)) + r202_col[None, :]), r202_value)
    }
    r203_z = tl.arange($(65536))
    r203_value = (tl.load($(candidate_vbmm_22_ptr) + ((((r203_z / $(512)) % $(128)) * $(512)) + (r203_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_23_ptr) + ((((r203_z / $(512)) % $(128)) * $(512)) + (r203_z % $(512))), dtype=tl.bfloat16))
    r203_value = r203_value.to(tl.bfloat16)
    r203_value = tl.broadcast_to(r203_value, [$(65536)])
    tl.store($(candidate_vadd_15_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(4096))
    r204_value = (tl.load($(candidate_vsum_4_ptr) + ((((r204_z / $(32)) % $(128)) * $(64)) + (r204_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r204_z / $(32)) % $(128)) * $(64)) + (r204_z % $(32))), dtype=tl.bfloat16))
    r204_value = r204_value.to(tl.bfloat16)
    r204_value = tl.broadcast_to(r204_value, [$(4096)])
    tl.store($(candidate_vmul_39_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(4096))
    r205_value = (tl.load($(candidate_vsum_4_ptr) + ((((r205_z / $(32)) % $(128)) * $(64)) + (r205_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r205_z / $(32)) % $(128)) * $(64)) + (r205_z % $(32))), dtype=tl.bfloat16))
    r205_value = r205_value.to(tl.bfloat16)
    r205_value = tl.broadcast_to(r205_value, [$(4096)])
    tl.store($(candidate_vmul_40_ptr) + r205_z, r205_value)
    r206_z = tl.arange($(4096))
    r206_value = -tl.load($(candidate_vsum_4_ptr) + ((((r206_z / $(32)) % $(128)) * $(64)) + (r206_z % $(32))), dtype=tl.bfloat16)
    r206_value = r206_value.to(tl.bfloat16)
    r206_value = tl.broadcast_to(r206_value, [$(4096)])
    tl.store($(candidate_vneg_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(4096))
    r207_value = (tl.load($(candidate_vneg_ptr) + ((((r207_z / $(32)) % $(128)) * $(32)) + (r207_z % $(32))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r207_z / $(32)) % $(128)) * $(64)) + (r207_z % $(32))), dtype=tl.bfloat16))
    r207_value = r207_value.to(tl.bfloat16)
    r207_value = tl.broadcast_to(r207_value, [$(4096)])
    tl.store($(candidate_vmul_41_ptr) + r207_z, r207_value)
    r208_z = tl.arange($(4096))
    r208_value = (tl.load($(candidate_vmul_40_ptr) + ((((r208_z / $(32)) % $(128)) * $(32)) + (r208_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_41_ptr) + ((((r208_z / $(32)) % $(128)) * $(32)) + (r208_z % $(32))), dtype=tl.bfloat16))
    r208_value = r208_value.to(tl.bfloat16)
    r208_value = tl.broadcast_to(r208_value, [$(4096)])
    tl.store($(candidate_vadd_16_ptr) + r208_z, r208_value)
    r209_z = tl.arange($(4096))
    r209_value = (tl.load($(candidate_vsum_4_ptr) + ((((r209_z / $(32)) % $(128)) * $(64)) + (r209_z % $(32))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r209_z / $(32)) % $(128)) * $(64)) + (r209_z % $(32))), dtype=tl.bfloat16))
    r209_value = r209_value.to(tl.bfloat16)
    r209_value = tl.broadcast_to(r209_value, [$(4096)])
    tl.store($(candidate_vmul_42_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(4096))
    r210_value = (tl.load($(candidate_vmul_39_ptr) + ((((r210_z / $(32)) % $(128)) * $(32)) + (r210_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_42_ptr) + ((((r210_z / $(32)) % $(128)) * $(32)) + (r210_z % $(32))), dtype=tl.bfloat16))
    r210_value = r210_value.to(tl.bfloat16)
    r210_value = tl.broadcast_to(r210_value, [$(4096)])
    tl.store($(candidate_vadd_17_ptr) + r210_z, r210_value)
    r211_z = tl.arange($(32768))
    r211_value = (tl.load($(candidate_vcat_10_ptr) + ((((r211_z / $(32)) % $(128)) * $(1536)) + (((r211_z / $(4096)) % $(8)) * $(192)) + (r211_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r211_z / $(32)) % $(128)) * $(64)) + (r211_z % $(32))), dtype=tl.bfloat16))
    r211_value = r211_value.to(tl.bfloat16)
    r211_value = tl.broadcast_to(r211_value, [$(32768)])
    tl.store($(candidate_vmul_43_ptr) + r211_z, r211_value)
    r212_z = tl.arange($(32768))
    r212_value = (tl.load($(candidate_vcat_10_ptr) + ((((r212_z / $(32)) % $(128)) * $(1536)) + (((r212_z / $(4096)) % $(8)) * $(192)) + (r212_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r212_z / $(32)) % $(128)) * $(64)) + (r212_z % $(32))), dtype=tl.bfloat16))
    r212_value = r212_value.to(tl.bfloat16)
    r212_value = tl.broadcast_to(r212_value, [$(32768)])
    tl.store($(candidate_vmul_44_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(32768))
    r213_value = -tl.load($(candidate_vcat_10_ptr) + ((((r213_z / $(32)) % $(128)) * $(1536)) + (((r213_z / $(4096)) % $(8)) * $(192)) + (r213_z % $(32)) + $(128)), dtype=tl.bfloat16)
    r213_value = r213_value.to(tl.bfloat16)
    r213_value = tl.broadcast_to(r213_value, [$(32768)])
    tl.store($(candidate_vneg_1_ptr) + r213_z, r213_value)
    r214_z = tl.arange($(32768))
    r214_value = (tl.load($(candidate_vneg_1_ptr) + ((((r214_z / $(4096)) % $(8)) * $(4096)) + (((r214_z / $(32)) % $(128)) * $(32)) + (r214_z % $(32))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r214_z / $(32)) % $(128)) * $(64)) + (r214_z % $(32))), dtype=tl.bfloat16))
    r214_value = r214_value.to(tl.bfloat16)
    r214_value = tl.broadcast_to(r214_value, [$(32768)])
    tl.store($(candidate_vmul_45_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(32768))
    r215_value = (tl.load($(candidate_vmul_44_ptr) + ((((r215_z / $(4096)) % $(8)) * $(4096)) + (((r215_z / $(32)) % $(128)) * $(32)) + (r215_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_45_ptr) + ((((r215_z / $(4096)) % $(8)) * $(4096)) + (((r215_z / $(32)) % $(128)) * $(32)) + (r215_z % $(32))), dtype=tl.bfloat16))
    r215_value = r215_value.to(tl.bfloat16)
    r215_value = tl.broadcast_to(r215_value, [$(32768)])
    tl.store($(candidate_vadd_18_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(32768))
    r216_value = (tl.load($(candidate_vcat_10_ptr) + ((((r216_z / $(32)) % $(128)) * $(1536)) + (((r216_z / $(4096)) % $(8)) * $(192)) + (r216_z % $(32)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r216_z / $(32)) % $(128)) * $(64)) + (r216_z % $(32))), dtype=tl.bfloat16))
    r216_value = r216_value.to(tl.bfloat16)
    r216_value = tl.broadcast_to(r216_value, [$(32768)])
    tl.store($(candidate_vmul_46_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(32768))
    r217_value = (tl.load($(candidate_vmul_43_ptr) + ((((r217_z / $(4096)) % $(8)) * $(4096)) + (((r217_z / $(32)) % $(128)) * $(32)) + (r217_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_46_ptr) + ((((r217_z / $(4096)) % $(8)) * $(4096)) + (((r217_z / $(32)) % $(128)) * $(32)) + (r217_z % $(32))), dtype=tl.bfloat16))
    r217_value = r217_value.to(tl.bfloat16)
    r217_value = tl.broadcast_to(r217_value, [$(32768)])
    tl.store($(candidate_vadd_19_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(8192))
    r218_value = tl.where((((r218_z % $(64)) >= $(1)) & ((r218_z % $(64)) < $(64)) & (((r218_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_16_ptr) + ((((r218_z / $(64)) % $(128)) * $(32)) + ((((r218_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r218_value = r218_value.to(tl.bfloat16)
    r218_value = tl.broadcast_to(r218_value, [$(8192)])
    tl.store($(candidate_vslice_backward_ptr) + r218_z, r218_value)
    r219_z = tl.arange($(8192))
    r219_value = tl.where((((r219_z % $(64)) >= $(0)) & ((r219_z % $(64)) < $(64)) & (((r219_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_17_ptr) + ((((r219_z / $(64)) % $(128)) * $(32)) + ((((r219_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r219_value = r219_value.to(tl.bfloat16)
    r219_value = tl.broadcast_to(r219_value, [$(8192)])
    tl.store($(candidate_vslice_backward_1_ptr) + r219_z, r219_value)
    r220_z = tl.arange($(8192))
    r220_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r220_z / $(64)) % $(128)) * $(64)) + (r220_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_1_ptr) + ((((r220_z / $(64)) % $(128)) * $(64)) + (r220_z % $(64))), dtype=tl.bfloat16))
    r220_value = r220_value.to(tl.bfloat16)
    r220_value = tl.broadcast_to(r220_value, [$(8192)])
    tl.store($(candidate_vadd_20_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(65536))
    r221_value = tl.where((((r221_z % $(64)) >= $(1)) & ((r221_z % $(64)) < $(64)) & (((r221_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_18_ptr) + ((((r221_z / $(8192)) % $(8)) * $(4096)) + (((r221_z / $(64)) % $(128)) * $(32)) + ((((r221_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r221_value = r221_value.to(tl.bfloat16)
    r221_value = tl.broadcast_to(r221_value, [$(65536)])
    tl.store($(candidate_vslice_backward_2_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(65536))
    r222_value = tl.where((((r222_z % $(64)) >= $(0)) & ((r222_z % $(64)) < $(64)) & (((r222_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_19_ptr) + ((((r222_z / $(8192)) % $(8)) * $(4096)) + (((r222_z / $(64)) % $(128)) * $(32)) + ((((r222_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r222_value = r222_value.to(tl.bfloat16)
    r222_value = tl.broadcast_to(r222_value, [$(65536)])
    tl.store($(candidate_vslice_backward_3_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(65536))
    r223_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r223_z / $(8192)) % $(8)) * $(8192)) + (((r223_z / $(64)) % $(128)) * $(64)) + (r223_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_3_ptr) + ((((r223_z / $(8192)) % $(8)) * $(8192)) + (((r223_z / $(64)) % $(128)) * $(64)) + (r223_z % $(64))), dtype=tl.bfloat16))
    r223_value = r223_value.to(tl.bfloat16)
    r223_value = tl.broadcast_to(r223_value, [$(65536)])
    tl.store($(candidate_vadd_21_ptr) + r223_z, r223_value)
    r224_z = tl.arange($(196608))
    r224_value = tl.where((r224_z % $(192)) < $(128), tl.load($(candidate_vcat_10_ptr) + ((((((((r224_z / $(192)) % $(128)) * $(1024)) + (((r224_z / $(24576)) % $(8)) * $(128)) + ((r224_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r224_z / $(192)) % $(128)) * $(1024)) + (((r224_z / $(24576)) % $(8)) * $(128)) + ((r224_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r224_z / $(192)) % $(128)) * $(1024)) + (((r224_z / $(24576)) % $(8)) * $(128)) + ((r224_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_21_ptr) + ((((r224_z / $(24576)) % $(8)) * $(8192)) + (((r224_z / $(192)) % $(128)) * $(64)) + (((r224_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r224_value = r224_value.to(tl.bfloat16)
    r224_value = tl.broadcast_to(r224_value, [$(196608)])
    tl.store($(candidate_vcat_14_ptr) + r224_z, r224_value)
    r225_z = tl.arange($(65536))
    r225_value = tl.load($(candidate_vadd_15_ptr) + ((((r225_z / $(512)) % $(128)) * $(512)) + (r225_z % $(512))), dtype=tl.bfloat16)
    r225_value = r225_value.to(tl.float32)
    r225_value = tl.broadcast_to(r225_value, [$(65536)])
    tl.store($(candidate_vto_copy_61_ptr) + r225_z, r225_value)
    r226_z = tl.arange($(65536))
    r226_value = (tl.load($(candidate_vto_copy_61_ptr) + ((((r226_z / $(512)) % $(128)) * $(512)) + (r226_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r226_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r226_value = r226_value.to(tl.float32)
    r226_value = tl.broadcast_to(r226_value, [$(65536)])
    tl.store($(candidate_vmul_47_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(65536))
    r227_value = (tl.load($(candidate_vmul_47_ptr) + ((((r227_z / $(512)) % $(128)) * $(512)) + (r227_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r227_z / $(512)) % $(128)) * $(512)) + (r227_z % $(512))), dtype=tl.float32))
    r227_value = r227_value.to(tl.float32)
    r227_value = tl.broadcast_to(r227_value, [$(65536)])
    tl.store($(candidate_vmul_48_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(65536))
    r228_value = (tl.load($(candidate_vmul_47_ptr) + ((((r228_z / $(512)) % $(128)) * $(512)) + (r228_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r228_z / $(512)) % $(128)), dtype=tl.float32))
    r228_value = r228_value.to(tl.float32)
    r228_value = tl.broadcast_to(r228_value, [$(65536)])
    tl.store($(candidate_vmul_49_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(128))
    r229_lane = tl.arange($(512))
    r229_value = tl.sum(tl.load($(candidate_vmul_48_ptr) + ((r229_z[:, None] * $(512)) + r229_lane[None, :]), dtype=tl.float32), axis=1)
    r229_value = r229_value.to(tl.float32)
    r229_value = tl.broadcast_to(r229_value, [$(128)])
    tl.store($(candidate_vsum_5_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(128))
    r230_value = ((tl.load($(candidate_vrsqrt_1_ptr) + r230_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r230_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r230_z, dtype=tl.float32)))
    r230_value = r230_value.to(tl.float32)
    r230_value = tl.broadcast_to(r230_value, [$(128)])
    tl.store($(candidate_vpow_4_ptr) + r230_z, r230_value)
    r231_z = tl.arange($(128))
    r231_value = (tl.load($(candidate_vsum_5_ptr) + r231_z, dtype=tl.float32) * -0.5)
    r231_value = r231_value.to(tl.float32)
    r231_value = tl.broadcast_to(r231_value, [$(128)])
    tl.store($(candidate_vmul_50_ptr) + r231_z, r231_value)
    r232_z = tl.arange($(128))
    r232_value = (tl.load($(candidate_vmul_50_ptr) + r232_z, dtype=tl.float32) * tl.load($(candidate_vpow_4_ptr) + r232_z, dtype=tl.float32))
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(128)])
    tl.store($(candidate_vmul_51_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(65536))
    r233_value = (tl.load($(candidate_vmul_51_ptr) + ((r233_z / $(512)) % $(128)), dtype=tl.float32) / 512.0)
    r233_value = r233_value.to(tl.float32)
    r233_value = tl.broadcast_to(r233_value, [$(65536)])
    tl.store($(candidate_vdiv_1_ptr) + r233_z, r233_value)
    r234_z = tl.arange($(65536))
    r234_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r234_z / $(512)) % $(128)) * $(512)) + (r234_z % $(512))), dtype=tl.float32)))
    r234_value = r234_value.to(tl.float32)
    r234_value = tl.broadcast_to(r234_value, [$(65536)])
    tl.store($(candidate_vpow_5_ptr) + r234_z, r234_value)
    r235_z = tl.arange($(65536))
    r235_value = (tl.load($(candidate_vpow_5_ptr) + ((((r235_z / $(512)) % $(128)) * $(512)) + (r235_z % $(512))), dtype=tl.float32) * 2.0)
    r235_value = r235_value.to(tl.float32)
    r235_value = tl.broadcast_to(r235_value, [$(65536)])
    tl.store($(candidate_vmul_52_ptr) + r235_z, r235_value)
    r236_z = tl.arange($(65536))
    r236_value = (tl.load($(candidate_vdiv_1_ptr) + ((((r236_z / $(512)) % $(128)) * $(512)) + (r236_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vmul_52_ptr) + ((((r236_z / $(512)) % $(128)) * $(512)) + (r236_z % $(512))), dtype=tl.float32))
    r236_value = r236_value.to(tl.float32)
    r236_value = tl.broadcast_to(r236_value, [$(65536)])
    tl.store($(candidate_vmul_53_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(65536))
    r237_value = (tl.load($(candidate_vmul_49_ptr) + ((((r237_z / $(512)) % $(128)) * $(512)) + (r237_z % $(512))), dtype=tl.float32) + tl.load($(candidate_vmul_53_ptr) + ((((r237_z / $(512)) % $(128)) * $(512)) + (r237_z % $(512))), dtype=tl.float32))
    r237_value = r237_value.to(tl.float32)
    r237_value = tl.broadcast_to(r237_value, [$(65536)])
    tl.store($(candidate_vadd_22_ptr) + r237_z, r237_value)
    r238_z = tl.arange($(65536))
    r238_value = tl.load($(candidate_vadd_22_ptr) + ((((r238_z / $(512)) % $(128)) * $(512)) + (r238_z % $(512))), dtype=tl.float32)
    r238_value = r238_value.to(tl.bfloat16)
    r238_value = tl.broadcast_to(r238_value, [$(65536)])
    tl.store($(candidate_vto_copy_62_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(73728))
    r239_value = tl.where((r239_z % $(576)) < $(512), tl.load($(candidate_vto_copy_62_ptr) + ((((r239_z / $(576)) % $(128)) * $(512)) + (((r239_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_20_ptr) + ((((((((r239_z / $(576)) % $(128)) * $(64)) + ((r239_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r239_z / $(576)) % $(128)) * $(64)) + ((r239_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r239_value = r239_value.to(tl.bfloat16)
    r239_value = tl.broadcast_to(r239_value, [$(73728)])
    tl.store($(candidate_vcat_15_ptr) + r239_z, r239_value)
    r240_row = tl.arange($(128))
    r240_col = tl.arange($(512))
    r240_lane = tl.arange($(576))
    r240_left = tl.load($(candidate_vcat_15_ptr) + ((((((r240_row[:, None] * $(576)) + r240_lane[None, :]) / $(576)) % $(128)) * $(576)) + (((r240_row[:, None] * $(576)) + r240_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r240_right = tl.load($(candidate_vto_copy_3_ptr) + (((r240_lane[:, None] % $(576)) * $(512)) + (r240_col[None, :] % $(512))), dtype=tl.bfloat16)
    r240_value = tl.dot(r240_left, r240_right)
    r240_value = r240_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_11_ptr) + ((r240_row[:, None] * $(512)) + r240_col[None, :]), r240_value)
    r241_row = tl.arange($(128))
    r241_col = tl.arange($(256))
    r241_lane = tl.arange($(1536))
    r241_left = tl.load($(candidate_vcat_14_ptr) + ((((((((((r241_row[:, None] * $(1536)) + r241_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r241_row[:, None] * $(1536)) + r241_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r241_row[:, None] * $(1536)) + r241_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r241_row[:, None] * $(1536)) + r241_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r241_row[:, None] * $(1536)) + r241_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r241_right = tl.load($(candidate_vto_copy_2_ptr) + (((r241_lane[:, None] % $(1536)) * $(256)) + (r241_col[None, :] % $(256))), dtype=tl.bfloat16)
    r241_value = tl.dot(r241_left, r241_right)
    r241_value = r241_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_12_ptr) + ((r241_row[:, None] * $(256)) + r241_col[None, :]), r241_value)
    r242_z = tl.arange($(32768))
    r242_value = tl.load($(candidate_vmm_12_ptr) + ((((r242_z / $(256)) % $(128)) * $(256)) + (r242_z % $(256))), dtype=tl.bfloat16)
    r242_value = r242_value.to(tl.float32)
    r242_value = tl.broadcast_to(r242_value, [$(32768)])
    tl.store($(candidate_vto_copy_63_ptr) + r242_z, r242_value)
    r243_z = tl.arange($(32768))
    r243_value = (tl.load($(candidate_vto_copy_63_ptr) + ((((r243_z / $(256)) % $(128)) * $(256)) + (r243_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r243_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r243_value = r243_value.to(tl.float32)
    r243_value = tl.broadcast_to(r243_value, [$(32768)])
    tl.store($(candidate_vmul_54_ptr) + r243_z, r243_value)
    r244_z = tl.arange($(32768))
    r244_value = (tl.load($(candidate_vmul_54_ptr) + ((((r244_z / $(256)) % $(128)) * $(256)) + (r244_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_18_ptr) + ((((r244_z / $(256)) % $(128)) * $(256)) + (r244_z % $(256))), dtype=tl.float32))
    r244_value = r244_value.to(tl.float32)
    r244_value = tl.broadcast_to(r244_value, [$(32768)])
    tl.store($(candidate_vmul_55_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(32768))
    r245_value = (tl.load($(candidate_vmul_54_ptr) + ((((r245_z / $(256)) % $(128)) * $(256)) + (r245_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r245_z / $(256)) % $(128)), dtype=tl.float32))
    r245_value = r245_value.to(tl.float32)
    r245_value = tl.broadcast_to(r245_value, [$(32768)])
    tl.store($(candidate_vmul_56_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(128))
    r246_lane = tl.arange($(256))
    r246_value = tl.sum(tl.load($(candidate_vmul_55_ptr) + ((r246_z[:, None] * $(256)) + r246_lane[None, :]), dtype=tl.float32), axis=1)
    r246_value = r246_value.to(tl.float32)
    r246_value = tl.broadcast_to(r246_value, [$(128)])
    tl.store($(candidate_vsum_6_ptr) + r246_z, r246_value)
    r247_z = tl.arange($(128))
    r247_value = ((tl.load($(candidate_vrsqrt_ptr) + r247_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r247_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r247_z, dtype=tl.float32)))
    r247_value = r247_value.to(tl.float32)
    r247_value = tl.broadcast_to(r247_value, [$(128)])
    tl.store($(candidate_vpow_6_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(128))
    r248_value = (tl.load($(candidate_vsum_6_ptr) + r248_z, dtype=tl.float32) * -0.5)
    r248_value = r248_value.to(tl.float32)
    r248_value = tl.broadcast_to(r248_value, [$(128)])
    tl.store($(candidate_vmul_57_ptr) + r248_z, r248_value)
    r249_z = tl.arange($(128))
    r249_value = (tl.load($(candidate_vmul_57_ptr) + r249_z, dtype=tl.float32) * tl.load($(candidate_vpow_6_ptr) + r249_z, dtype=tl.float32))
    r249_value = r249_value.to(tl.float32)
    r249_value = tl.broadcast_to(r249_value, [$(128)])
    tl.store($(candidate_vmul_58_ptr) + r249_z, r249_value)
    r250_z = tl.arange($(32768))
    r250_value = (tl.load($(candidate_vmul_58_ptr) + ((r250_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r250_value = r250_value.to(tl.float32)
    r250_value = tl.broadcast_to(r250_value, [$(32768)])
    tl.store($(candidate_vdiv_2_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(32768))
    r251_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r251_z / $(256)) % $(128)) * $(256)) + (r251_z % $(256))), dtype=tl.float32)))
    r251_value = r251_value.to(tl.float32)
    r251_value = tl.broadcast_to(r251_value, [$(32768)])
    tl.store($(candidate_vpow_7_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(32768))
    r252_value = (tl.load($(candidate_vpow_7_ptr) + ((((r252_z / $(256)) % $(128)) * $(256)) + (r252_z % $(256))), dtype=tl.float32) * 2.0)
    r252_value = r252_value.to(tl.float32)
    r252_value = tl.broadcast_to(r252_value, [$(32768)])
    tl.store($(candidate_vmul_59_ptr) + r252_z, r252_value)
    r253_z = tl.arange($(32768))
    r253_value = (tl.load($(candidate_vdiv_2_ptr) + ((((r253_z / $(256)) % $(128)) * $(256)) + (r253_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_59_ptr) + ((((r253_z / $(256)) % $(128)) * $(256)) + (r253_z % $(256))), dtype=tl.float32))
    r253_value = r253_value.to(tl.float32)
    r253_value = tl.broadcast_to(r253_value, [$(32768)])
    tl.store($(candidate_vmul_60_ptr) + r253_z, r253_value)
    r254_z = tl.arange($(32768))
    r254_value = (tl.load($(candidate_vmul_56_ptr) + ((((r254_z / $(256)) % $(128)) * $(256)) + (r254_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_60_ptr) + ((((r254_z / $(256)) % $(128)) * $(256)) + (r254_z % $(256))), dtype=tl.float32))
    r254_value = r254_value.to(tl.float32)
    r254_value = tl.broadcast_to(r254_value, [$(32768)])
    tl.store($(candidate_vadd_23_ptr) + r254_z, r254_value)
    r255_z = tl.arange($(32768))
    r255_value = tl.load($(candidate_vadd_23_ptr) + ((((r255_z / $(256)) % $(128)) * $(256)) + (r255_z % $(256))), dtype=tl.float32)
    r255_value = r255_value.to(tl.bfloat16)
    r255_value = tl.broadcast_to(r255_value, [$(32768)])
    tl.store($(candidate_vto_copy_64_ptr) + r255_z, r255_value)
    r256_row = tl.arange($(128))
    r256_col = tl.arange($(512))
    r256_lane = tl.arange($(256))
    r256_left = tl.load($(candidate_vto_copy_64_ptr) + ((((((r256_row[:, None] * $(256)) + r256_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r256_row[:, None] * $(256)) + r256_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r256_right = tl.load($(candidate_vto_copy_ptr) + (((r256_lane[:, None] % $(256)) * $(512)) + (r256_col[None, :] % $(512))), dtype=tl.bfloat16)
    r256_value = tl.dot(r256_left, r256_right)
    r256_value = r256_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_13_ptr) + ((r256_row[:, None] * $(512)) + r256_col[None, :]), r256_value)
    r257_z = tl.arange($(65536))
    r257_value = (tl.load($(candidate_vmm_11_ptr) + ((((r257_z / $(512)) % $(128)) * $(512)) + (r257_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_13_ptr) + ((((r257_z / $(512)) % $(128)) * $(512)) + (r257_z % $(512))), dtype=tl.bfloat16))
    r257_value = r257_value.to(tl.bfloat16)
    r257_value = tl.broadcast_to(r257_value, [$(65536)])
    tl.store($(candidate_vadd_24_ptr) + r257_z, r257_value)
    z = tl.arange($(65536))
    value = tl.load($(candidate_vadd_24_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_reuse_varlen_dx
