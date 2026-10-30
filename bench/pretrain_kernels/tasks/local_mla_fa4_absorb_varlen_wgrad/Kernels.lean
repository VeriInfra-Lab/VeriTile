import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_varlen_wgrad
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
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vcat_8_ptr : RegionName := "vcat_8"
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
    r20_z = tl.arange($(4096))
    r20_value = (tl.cast(tl.toReal(((r20_z / $(32)) % $(128))), tl.float32) * tl.load($(vmul_7_ptr) + (r20_z % $(32)), dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(4096)])
    tl.store($(vmul_8_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(8192))
    r21_value = tl.where((r21_z % $(64)) < $(32), tl.load($(vmul_8_ptr) + ((((r21_z / $(64)) % $(128)) * $(32)) + (((r21_z % $(64)) - $(0)) % $(32))), dtype=tl.float32), tl.load($(vmul_8_ptr) + ((((r21_z / $(64)) % $(128)) * $(32)) + (((r21_z % $(64)) - $(32)) % $(32))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(8192)])
    tl.store($(vcat_1_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(8192))
    r22_value = tl.cos(tl.load($(vcat_1_ptr) + ((((r22_z / $(64)) % $(128)) * $(64)) + (r22_z % $(64))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(8192)])
    tl.store($(vcos_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(8192))
    r23_value = tl.load($(vcos_ptr) + (((((((tl.load($(vcat_ptr) + ((r23_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r23_z % $(64))) / $(64)) % $(128)) % $(128)) * $(64)) + ((((tl.load($(vcat_ptr) + ((r23_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r23_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(8192)])
    tl.store($(vindex_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(8192))
    r24_value = tl.sin(tl.load($(vcat_1_ptr) + ((((r24_z / $(64)) % $(128)) * $(64)) + (r24_z % $(64))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(8192)])
    tl.store($(vsin_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(8192))
    r25_value = tl.load($(vsin_ptr) + (((((((tl.load($(vcat_ptr) + ((r25_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r25_z % $(64))) / $(64)) % $(128)) % $(128)) * $(64)) + ((((tl.load($(vcat_ptr) + ((r25_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r25_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(8192)])
    tl.store($(vindex_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(8192))
    r26_value = tl.load($(vindex_ptr) + ((((r26_z / $(64)) % $(128)) * $(64)) + (r26_z % $(64))), dtype=tl.float32)
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(8192)])
    tl.store($(vto_copy_14_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(8192))
    r27_value = tl.load($(vindex_1_ptr) + ((((r27_z / $(64)) % $(128)) * $(64)) + (r27_z % $(64))), dtype=tl.float32)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(8192)])
    tl.store($(vto_copy_15_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(1))
    r28_value = (tl.load($(vcat_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r28_value = tl.broadcast_to(r28_value, [$(1)])
    tl.store($(vsub_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(129))
    r29_value = tl.where(r29_z < $(1), tl.load($(vsub_ptr) + $(0), dtype=tl.int64), tl.load($(vcat_ptr) + ((r29_z - $(1)) % $(128)), dtype=tl.int64))
    r29_value = tl.broadcast_to(r29_value, [$(129)])
    tl.store($(vcat_2_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(128))
    r30_value = (tl.load($(vcat_2_ptr) + (r30_z + $(1)), dtype=tl.int64) - tl.load($(vcat_2_ptr) + r30_z, dtype=tl.int64))
    r30_value = tl.broadcast_to(r30_value, [$(128)])
    tl.store($(vsub_1_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(128))
    r31_value = (tl.load($(vsub_1_ptr) + r31_z, dtype=tl.int64) != tl.full([], $(1), dtype=tl.int64))
    r31_value = tl.broadcast_to(r31_value, [$(128)])
    tl.store($(vne_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(128))
    r32_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(128)])
    for r32_lane in range($(128)) {
      r32_value = tl.where(r32_lane <= r32_z, (r32_value + tl.where(tl.load($(vne_ptr) + (r32_lane % $(128)), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))), r32_value)
    }
    r32_value = tl.broadcast_to(r32_value, [$(128)])
    tl.store($(vcumsum_1_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(16384))
    r33_value = (tl.load($(vcumsum_1_ptr) + ((r33_z / $(128)) % $(128)), dtype=tl.int64) == tl.load($(vcumsum_1_ptr) + (r33_z % $(128)), dtype=tl.int64))
    r33_value = tl.broadcast_to(r33_value, [$(16384)])
    tl.store($(veq_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(16384))
    r34_value = tl.where((r34_z % $(128)) <= ((r34_z / $(128)) % $(128)) + $(0), true, false)
    r34_value = tl.broadcast_to(r34_value, [$(16384)])
    tl.store($(vtril_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(16384))
    r35_value = (tl.load($(veq_ptr) + ((((r35_z / $(128)) % $(128)) * $(128)) + (r35_z % $(128))), dtype=tl.int1) & tl.load($(vtril_ptr) + ((((r35_z / $(128)) % $(128)) * $(128)) + (r35_z % $(128))), dtype=tl.int1))
    r35_value = tl.broadcast_to(r35_value, [$(16384)])
    tl.store($(vbitwise_and_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(16384))
    r36_value = tl.where(tl.load($(vbitwise_and_ptr) + ((((r36_z / $(128)) % $(128)) * $(128)) + (r36_z % $(128))), dtype=tl.int1), tl.cast(0.0, tl.bfloat16), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16))
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(16384)])
    tl.store($(vwhere_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(1048576))
    r37_value = tl.load($(vmul_5_ptr) + ((((r37_z / $(512)) % $(2048)) * $(512)) + (r37_z % $(512))), dtype=tl.float32)
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(1048576)])
    tl.store($(vto_copy_18_ptr) + r37_z, r37_value)
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
    tl.store($(vto_copy_19_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(32768))
    r40_value = ((tl.load($(vto_copy_19_ptr) + ((((r40_z / $(256)) % $(128)) * $(256)) + (r40_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_19_ptr) + ((((r40_z / $(256)) % $(128)) * $(256)) + (r40_z % $(256))), dtype=tl.float32)))
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
    r44_value = (tl.load($(vto_copy_19_ptr) + ((((r44_z / $(256)) % $(128)) * $(256)) + (r44_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((r44_z / $(256)) % $(128)), dtype=tl.float32))
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
    tl.store($(vto_copy_20_ptr) + r46_z, r46_value)
    r47_row = tl.arange($(128))
    r47_col = tl.arange($(1536))
    r47_lane = tl.arange($(256))
    r47_left = tl.load($(vto_copy_20_ptr) + ((((((r47_row[:, None] * $(256)) + r47_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r47_row[:, None] * $(256)) + r47_lane[None, :]) % $(256))), dtype=tl.bfloat16)
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
    tl.store($(vto_copy_21_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(65536))
    r50_value = ((tl.load($(vto_copy_21_ptr) + ((((r50_z / $(512)) % $(128)) * $(512)) + (r50_z % $(512))), dtype=tl.float32)) * (tl.load($(vto_copy_21_ptr) + ((((r50_z / $(512)) % $(128)) * $(512)) + (r50_z % $(512))), dtype=tl.float32)))
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
    r54_value = (tl.load($(vto_copy_21_ptr) + ((((r54_z / $(512)) % $(128)) * $(512)) + (r54_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r54_z / $(512)) % $(128)), dtype=tl.float32))
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
    tl.store($(vto_copy_22_ptr) + r56_z, r56_value)
    r57_row = tl.arange($(128))
    r57_col = tl.arange($(2048))
    r57_lane = tl.arange($(512))
    r57_left = tl.load($(vto_copy_22_ptr) + ((((((r57_row[:, None] * $(512)) + r57_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r57_row[:, None] * $(512)) + r57_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r57_right = tl.load($(vto_copy_18_ptr) + (((r57_col[None, :] % $(2048)) * $(512)) + (r57_lane[:, None] % $(512))), dtype=tl.bfloat16)
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
    tl.store($(vto_copy_23_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(131072))
    r78_lane = tl.arange($(128))
    r78_row_values = tl.load($(vto_copy_23_ptr) + ((((r78_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r78_z[:, None] / $(128)) % $(128)) * $(128)) + (r78_lane[None, :] % $(128))), dtype=tl.float32)
    r78_maximum = tl.max(r78_row_values, axis=1, keep_dims=true)
    r78_weights = tl.exp(r78_row_values - r78_maximum)
    r78_total = tl.sum(r78_weights, axis=1)
    r78_maximum = tl.max(r78_row_values, axis=1)
    r78_value = tl.exp(tl.load($(vto_copy_23_ptr) + ((((r78_z / $(16384)) % $(8)) * $(16384)) + (((r78_z / $(128)) % $(128)) * $(128)) + (r78_z % $(128))), dtype=tl.float32) - r78_maximum) / r78_total
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(131072)])
    tl.store($(vsoftmax_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(131072))
    r79_value = tl.load($(vsoftmax_ptr) + ((((r79_z / $(16384)) % $(8)) * $(16384)) + (((r79_z / $(128)) % $(128)) * $(128)) + (r79_z % $(128))), dtype=tl.float32)
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(131072)])
    tl.store($(vto_copy_24_ptr) + r79_z, r79_value)
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
      r81_left = tl.load($(vto_copy_24_ptr) + (((((((r81_group % $(8)) * $(16384)) + (r81_lane[None, :] * $(128)) + r81_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r81_group % $(8)) * $(16384)) + (r81_lane[None, :] * $(128)) + r81_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r81_group % $(8)) * $(16384)) + (r81_lane[None, :] * $(128)) + r81_row[:, None]) % $(128))), dtype=tl.bfloat16)
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
    tl.store($(vto_copy_25_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(131072))
    r84_lane = tl.arange($(128))
    r84_row_values = tl.load($(vsoftmax_ptr) + ((((r84_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r84_z[:, None] / $(128)) % $(128)) * $(128)) + (r84_lane[None, :] % $(128))), dtype=tl.float32)
    r84_row_grad = tl.load($(vto_copy_25_ptr) + ((((r84_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r84_z[:, None] / $(128)) % $(128)) * $(128)) + (r84_lane[None, :] % $(128))), dtype=tl.float32)
    r84_weighted = r84_row_grad * r84_row_values
    r84_dot_grad = tl.sum(r84_weighted, axis=1)
    r84_value = (tl.load($(vto_copy_25_ptr) + ((((r84_z / $(16384)) % $(8)) * $(16384)) + (((r84_z / $(128)) % $(128)) * $(128)) + (r84_z % $(128))), dtype=tl.float32) - r84_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r84_z / $(16384)) % $(8)) * $(16384)) + (((r84_z / $(128)) % $(128)) * $(128)) + (r84_z % $(128))), dtype=tl.float32)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(131072)])
    tl.store($(vsoftmax_backward_data_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(131072))
    r85_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r85_z / $(16384)) % $(8)) * $(16384)) + (((r85_z / $(128)) % $(128)) * $(128)) + (r85_z % $(128))), dtype=tl.float32)
    r85_value = r85_value.to(tl.bfloat16)
    r85_value = tl.broadcast_to(r85_value, [$(131072)])
    tl.store($(vto_copy_26_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(131072))
    r86_value = (tl.load($(vto_copy_26_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(128)) % $(128)) * $(128)) + (r86_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
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
    r88_z = tl.arange($(262144))
    r88_value = tl.where((r88_z % $(256)) < $(128), tl.load($(vbmm_4_ptr) + ((((((((r88_z / $(32768)) % $(8)) * $(24576)) + (((r88_z % $(256)) - $(0)) * $(128)) + ((r88_z / $(256)) % $(128))) / $(24576)) % $(8)) * $(24576)) + ((((((r88_z / $(32768)) % $(8)) * $(192)) + ((r88_z % $(256)) - $(0))) % $(192)) * $(128)) + ((r88_z / $(256)) % $(128))), dtype=tl.bfloat16), tl.load($(vbmm_2_ptr) + ((((((((r88_z / $(32768)) % $(8)) * $(16384)) + (((r88_z / $(256)) % $(128)) * $(128)) + ((r88_z % $(256)) - $(128))) / $(16384)) % $(8)) * $(16384)) + (((((((r88_z / $(32768)) % $(8)) * $(16384)) + (((r88_z / $(256)) % $(128)) * $(128)) + ((r88_z % $(256)) - $(128))) / $(128)) % $(128)) * $(128)) + (((((r88_z / $(32768)) % $(8)) * $(16384)) + (((r88_z / $(256)) % $(128)) * $(128)) + ((r88_z % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16))
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(262144)])
    tl.store($(vcat_8_ptr) + r88_z, r88_value)
    r89_row = tl.arange($(2048))
    r89_col = tl.arange($(512))
    r89_lane = tl.arange($(128))
    r89_left = tl.load($(vcat_8_ptr) + ((((((((((r89_lane[None, :] * $(2048)) + r89_row[:, None]) / $(2048)) % $(128)) * $(2048)) + (((r89_lane[None, :] * $(2048)) + r89_row[:, None]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r89_lane[None, :] * $(2048)) + r89_row[:, None]) / $(2048)) % $(128)) * $(256)) + (((((((r89_lane[None, :] * $(2048)) + r89_row[:, None]) / $(2048)) % $(128)) * $(2048)) + (((r89_lane[None, :] * $(2048)) + r89_row[:, None]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r89_right = tl.load($(vto_copy_22_ptr) + ((((((r89_lane[:, None] * $(512)) + r89_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r89_lane[:, None] * $(512)) + r89_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r89_value = tl.dot(r89_left, r89_right)
    r89_value = r89_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r89_row[:, None] * $(512)) + r89_col[None, :]), r89_value)
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
  let vcat_7_ptr : RegionName := "vcat_7"
  let vcat_8_ptr : RegionName := "vcat_8"
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
  let candidate_vmm_10_ptr : RegionName := "candidate_vmm_10"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vsub_9_ptr : RegionName := "candidate_vsub_9"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vlogsumexp_3_ptr : RegionName := "candidate_vlogsumexp_3"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_10_ptr : RegionName := "candidate_vsub_10"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsoftmax_4_ptr : RegionName := "candidate_vsoftmax_4"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vlogsumexp_4_ptr : RegionName := "candidate_vlogsumexp_4"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsoftmax_5_ptr : RegionName := "candidate_vsoftmax_5"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vlogsumexp_5_ptr : RegionName := "candidate_vlogsumexp_5"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vsub_12_ptr : RegionName := "candidate_vsub_12"
  let candidate_vgt_6_ptr : RegionName := "candidate_vgt_6"
  let candidate_vmasked_fill_6_ptr : RegionName := "candidate_vmasked_fill_6"
  let candidate_vsub_13_ptr : RegionName := "candidate_vsub_13"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vsub_14_ptr : RegionName := "candidate_vsub_14"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vto_copy_56_ptr : RegionName := "candidate_vto_copy_56"
  let candidate_vto_copy_57_ptr : RegionName := "candidate_vto_copy_57"
  let candidate_vto_copy_58_ptr : RegionName := "candidate_vto_copy_58"
  let candidate_vto_copy_59_ptr : RegionName := "candidate_vto_copy_59"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vgt_7_ptr : RegionName := "candidate_vgt_7"
  let candidate_vmasked_fill_7_ptr : RegionName := "candidate_vmasked_fill_7"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vbmm_21_ptr : RegionName := "candidate_vbmm_21"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vto_copy_61_ptr : RegionName := "candidate_vto_copy_61"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vto_copy_62_ptr : RegionName := "candidate_vto_copy_62"
  let candidate_vto_copy_63_ptr : RegionName := "candidate_vto_copy_63"
  let candidate_vto_copy_64_ptr : RegionName := "candidate_vto_copy_64"
  let candidate_vto_copy_65_ptr : RegionName := "candidate_vto_copy_65"
  let candidate_vto_copy_66_ptr : RegionName := "candidate_vto_copy_66"
  let candidate_vto_copy_67_ptr : RegionName := "candidate_vto_copy_67"
  let candidate_vbmm_23_ptr : RegionName := "candidate_vbmm_23"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vsub_18_ptr : RegionName := "candidate_vsub_18"
  let candidate_vgt_8_ptr : RegionName := "candidate_vgt_8"
  let candidate_vmasked_fill_8_ptr : RegionName := "candidate_vmasked_fill_8"
  let candidate_vsub_19_ptr : RegionName := "candidate_vsub_19"
  let candidate_vexp_2_ptr : RegionName := "candidate_vexp_2"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vbmm_24_ptr : RegionName := "candidate_vbmm_24"
  let candidate_vsub_20_ptr : RegionName := "candidate_vsub_20"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vbmm_26_ptr : RegionName := "candidate_vbmm_26"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vto_copy_69_ptr : RegionName := "candidate_vto_copy_69"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vto_copy_70_ptr : RegionName := "candidate_vto_copy_70"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vcat_13_ptr : RegionName := "candidate_vcat_13"
  let candidate_vbmm_30_ptr : RegionName := "candidate_vbmm_30"
  let candidate_vbmm_31_ptr : RegionName := "candidate_vbmm_31"
  let candidate_vcat_14_ptr : RegionName := "candidate_vcat_14"
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
    r20_z = tl.arange($(4096))
    r20_value = (tl.cast(tl.toReal(((r20_z / $(32)) % $(128))), tl.float32) * tl.load($(candidate_vmul_7_ptr) + (r20_z % $(32)), dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(4096)])
    tl.store($(candidate_vmul_8_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(8192))
    r21_value = tl.where((r21_z % $(64)) < $(32), tl.load($(candidate_vmul_8_ptr) + ((((r21_z / $(64)) % $(128)) * $(32)) + (((r21_z % $(64)) - $(0)) % $(32))), dtype=tl.float32), tl.load($(candidate_vmul_8_ptr) + ((((r21_z / $(64)) % $(128)) * $(32)) + (((r21_z % $(64)) - $(32)) % $(32))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(8192)])
    tl.store($(candidate_vcat_1_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(8192))
    r22_value = tl.cos(tl.load($(candidate_vcat_1_ptr) + ((((r22_z / $(64)) % $(128)) * $(64)) + (r22_z % $(64))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(8192)])
    tl.store($(candidate_vcos_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(8192))
    r23_value = tl.load($(candidate_vcos_ptr) + (((((((tl.load($(candidate_vcat_ptr) + ((r23_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r23_z % $(64))) / $(64)) % $(128)) % $(128)) * $(64)) + ((((tl.load($(candidate_vcat_ptr) + ((r23_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r23_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(8192)])
    tl.store($(candidate_vindex_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(8192))
    r24_value = tl.sin(tl.load($(candidate_vcat_1_ptr) + ((((r24_z / $(64)) % $(128)) * $(64)) + (r24_z % $(64))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(8192)])
    tl.store($(candidate_vsin_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(8192))
    r25_value = tl.load($(candidate_vsin_ptr) + (((((((tl.load($(candidate_vcat_ptr) + ((r25_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r25_z % $(64))) / $(64)) % $(128)) % $(128)) * $(64)) + ((((tl.load($(candidate_vcat_ptr) + ((r25_z / $(64)) % $(128)), dtype=tl.int64) * $(64)) + (r25_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(8192)])
    tl.store($(candidate_vindex_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(8192))
    r26_value = tl.load($(candidate_vindex_ptr) + ((((r26_z / $(64)) % $(128)) * $(64)) + (r26_z % $(64))), dtype=tl.float32)
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(8192)])
    tl.store($(candidate_vto_copy_14_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(8192))
    r27_value = tl.load($(candidate_vindex_1_ptr) + ((((r27_z / $(64)) % $(128)) * $(64)) + (r27_z % $(64))), dtype=tl.float32)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(8192)])
    tl.store($(candidate_vto_copy_15_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(1048576))
    r28_value = tl.load($(candidate_vmul_5_ptr) + ((((r28_z / $(512)) % $(2048)) * $(512)) + (r28_z % $(512))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(1048576)])
    tl.store($(candidate_vto_copy_18_ptr) + r28_z, r28_value)
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
    tl.store($(candidate_vto_copy_19_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(32768))
    r31_value = ((tl.load($(candidate_vto_copy_19_ptr) + ((((r31_z / $(256)) % $(128)) * $(256)) + (r31_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_19_ptr) + ((((r31_z / $(256)) % $(128)) * $(256)) + (r31_z % $(256))), dtype=tl.float32)))
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
    r35_value = (tl.load($(candidate_vto_copy_19_ptr) + ((((r35_z / $(256)) % $(128)) * $(256)) + (r35_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r35_z / $(256)) % $(128)), dtype=tl.float32))
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
    tl.store($(candidate_vto_copy_20_ptr) + r37_z, r37_value)
    r38_row = tl.arange($(128))
    r38_col = tl.arange($(1536))
    r38_lane = tl.arange($(256))
    r38_left = tl.load($(candidate_vto_copy_20_ptr) + ((((((r38_row[:, None] * $(256)) + r38_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r38_row[:, None] * $(256)) + r38_lane[None, :]) % $(256))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_21_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(65536))
    r41_value = ((tl.load($(candidate_vto_copy_21_ptr) + ((((r41_z / $(512)) % $(128)) * $(512)) + (r41_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_21_ptr) + ((((r41_z / $(512)) % $(128)) * $(512)) + (r41_z % $(512))), dtype=tl.float32)))
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
    r45_value = (tl.load($(candidate_vto_copy_21_ptr) + ((((r45_z / $(512)) % $(128)) * $(512)) + (r45_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r45_z / $(512)) % $(128)), dtype=tl.float32))
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
    tl.store($(candidate_vto_copy_22_ptr) + r47_z, r47_value)
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
    r62_col = tl.arange($(1024))
    r62_lane = tl.arange($(512))
    r62_left = tl.load($(input1_ptr) + ((((((r62_row[:, None] * $(512)) + r62_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r62_row[:, None] * $(512)) + r62_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r62_right = tl.load($(candidate_vto_copy_6_ptr) + (((r62_lane[:, None] % $(512)) * $(1024)) + (r62_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r62_value = tl.dot(r62_left, r62_right)
    r62_value = r62_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_10_ptr) + ((r62_row[:, None] * $(1024)) + r62_col[None, :]), r62_value)
    r63_row = tl.arange($(128))
    r63_col = tl.arange($(1024))
    r63_lane = tl.arange($(512))
    for r63_group in range($(1)) {
      r63_left = tl.load($(candidate_vto_copy_22_ptr) + ((((((r63_row[:, None] * $(512)) + r63_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r63_row[:, None] * $(512)) + r63_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r63_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r63_lane[:, None] * $(1024)) + r63_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r63_lane[:, None] * $(1024)) + r63_col[None, :]) % $(128)) * $(512)) + ((((r63_lane[:, None] * $(1024)) + r63_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r63_value = tl.dot(r63_left, r63_right)
      r63_value = r63_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r63_group * $(131072)) + (r63_row[:, None] * $(1024)) + r63_col[None, :]), r63_value)
    }
    r64_row = tl.arange($(128))
    r64_col = tl.arange($(1024))
    r64_lane = tl.arange($(512))
    for r64_group in range($(1)) {
      r64_left = tl.load($(candidate_vto_copy_22_ptr) + ((((((r64_row[:, None] * $(512)) + r64_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r64_row[:, None] * $(512)) + r64_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r64_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r64_lane[:, None] * $(1024)) + r64_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r64_lane[:, None] * $(1024)) + r64_col[None, :]) % $(128)) * $(512)) + ((((r64_lane[:, None] * $(1024)) + r64_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r64_value = tl.dot(r64_left, r64_right)
      r64_value = r64_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r64_group * $(131072)) + (r64_row[:, None] * $(1024)) + r64_col[None, :]), r64_value)
    }
    r65_z = tl.arange($(196608))
    r65_value = tl.where((r65_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r65_z / $(1536)) % $(128)) * $(1024)) + (((r65_z / $(192)) % $(8)) * $(128)) + ((r65_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r65_z / $(1536)) % $(128)) * $(1024)) + (((r65_z / $(192)) % $(8)) * $(128)) + ((r65_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r65_z / $(1536)) % $(128)) * $(1024)) + (((r65_z / $(192)) % $(8)) * $(128)) + ((r65_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_4_ptr) + ((((((((r65_z / $(1536)) % $(128)) * $(512)) + (((r65_z / $(192)) % $(8)) * $(64)) + ((r65_z % $(192)) - $(128))) / $(64)) % $(8)) * $(8192)) + (((((((r65_z / $(1536)) % $(128)) * $(512)) + (((r65_z / $(192)) % $(8)) * $(64)) + ((r65_z % $(192)) - $(128))) / $(512)) % $(128)) * $(64)) + (((((r65_z / $(1536)) % $(128)) * $(512)) + (((r65_z / $(192)) % $(8)) * $(64)) + ((r65_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(196608)])
    tl.store($(candidate_vcat_7_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(196608))
    r66_value = tl.where((r66_z % $(192)) < $(128), tl.load($(candidate_vbmm_5_ptr) + ((((((((r66_z / $(1536)) % $(128)) * $(1024)) + (((r66_z / $(192)) % $(8)) * $(128)) + ((r66_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r66_z / $(1536)) % $(128)) * $(1024)) + (((r66_z / $(192)) % $(8)) * $(128)) + ((r66_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r66_z / $(1536)) % $(128)) * $(1024)) + (((r66_z / $(192)) % $(8)) * $(128)) + ((r66_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r66_z / $(1536)) % $(128)) * $(64)) + ((r66_z % $(192)) - $(128))) / $(64)) % $(128)) * $(64)) + (((((r66_z / $(1536)) % $(128)) * $(64)) + ((r66_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(196608)])
    tl.store($(candidate_vcat_8_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(73728))
    r67_value = tl.load($(candidate_vcat_7_ptr) + ((((r67_z / $(192)) % $(48)) * $(1536)) + (((r67_z / $(9216)) % $(8)) * $(192)) + (r67_z % $(192))), dtype=tl.bfloat16)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(73728)])
    tl.store($(candidate_vto_copy_38_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(73728))
    r68_value = tl.load($(candidate_vcat_8_ptr) + ((((r68_z / $(192)) % $(48)) * $(1536)) + (((r68_z / $(9216)) % $(8)) * $(192)) + (r68_z % $(192))), dtype=tl.bfloat16)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(73728)])
    tl.store($(candidate_vto_copy_39_ptr) + r68_z, r68_value)
    r69_row = tl.arange($(48))
    r69_col = tl.arange($(48))
    r69_lane = tl.arange($(192))
    for r69_group in range($(8)) {
      r69_left = tl.load($(candidate_vto_copy_38_ptr) + (((((((r69_group % $(8)) * $(9216)) + (r69_row[:, None] * $(192)) + r69_lane[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r69_group % $(8)) * $(9216)) + (r69_row[:, None] * $(192)) + r69_lane[None, :]) / $(192)) % $(48)) * $(192)) + ((((r69_group % $(8)) * $(9216)) + (r69_row[:, None] * $(192)) + r69_lane[None, :]) % $(192))), dtype=tl.float32)
      r69_right = tl.load($(candidate_vto_copy_39_ptr) + (((((((r69_group % $(8)) * $(9216)) + (r69_lane[:, None] * $(48)) + r69_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r69_group % $(8)) * $(9216)) + (r69_lane[:, None] * $(48)) + r69_col[None, :]) % $(48)) * $(192)) + (((((r69_group % $(8)) * $(9216)) + (r69_lane[:, None] * $(48)) + r69_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r69_value = tl.dot(r69_left, r69_right)
      r69_value = r69_value.to(tl.float32)
      tl.store($(candidate_vbmm_7_ptr) + ((r69_group * $(2304)) + (r69_row[:, None] * $(48)) + r69_col[None, :]), r69_value)
    }
    r70_z = tl.arange($(18432))
    r70_value = (tl.load($(candidate_vbmm_7_ptr) + ((((r70_z / $(2304)) % $(8)) * $(2304)) + (((r70_z / $(48)) % $(48)) * $(48)) + (r70_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(18432)])
    tl.store($(candidate_vmul_24_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(48))
    r71_value = (tl.cast(r71_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r71_value = tl.broadcast_to(r71_value, [$(48)])
    tl.store($(candidate_vadd_12_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(48))
    r72_value = (tl.load($(candidate_vadd_12_ptr) + r72_z, dtype=tl.int64) - tl.full([], $(48), dtype=tl.int64))
    r72_value = tl.broadcast_to(r72_value, [$(48)])
    tl.store($(candidate_vsub_9_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(2304))
    r73_value = (tl.cast((r73_z % $(48)), tl.int64) > tl.load($(candidate_vsub_9_ptr) + ((r73_z / $(48)) % $(48)), dtype=tl.int64))
    r73_value = tl.broadcast_to(r73_value, [$(2304)])
    tl.store($(candidate_vgt_3_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(18432))
    r74_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r74_z / $(48)) % $(48)) * $(48)) + (r74_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r74_z / $(2304)) % $(8)) * $(2304)) + (((r74_z / $(48)) % $(48)) * $(48)) + (r74_z % $(48))), dtype=tl.float32))
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(18432))
    r75_lane = tl.arange($(48))
    r75_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r75_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r75_z[:, None] / $(48)) % $(48)) * $(48)) + (r75_lane[None, :] % $(48))), dtype=tl.float32)
    r75_maximum = tl.max(r75_row_values, axis=1, keep_dims=true)
    r75_weights = tl.exp(r75_row_values - r75_maximum)
    r75_total = tl.sum(r75_weights, axis=1)
    r75_maximum = tl.max(r75_row_values, axis=1)
    r75_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r75_z / $(2304)) % $(8)) * $(2304)) + (((r75_z / $(48)) % $(48)) * $(48)) + (r75_z % $(48))), dtype=tl.float32) - r75_maximum) / r75_total
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(18432)])
    tl.store($(candidate_vsoftmax_3_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(18432))
    r76_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r76_z / $(2304)) % $(8)) * $(2304)) + (((r76_z / $(48)) % $(48)) * $(48)) + (r76_z % $(48))), dtype=tl.float32)
    r76_value = r76_value.to(tl.bfloat16)
    r76_value = tl.broadcast_to(r76_value, [$(18432)])
    tl.store($(candidate_vto_copy_40_ptr) + r76_z, r76_value)
    r77_row = tl.arange($(48))
    r77_col = tl.arange($(128))
    r77_lane = tl.arange($(48))
    for r77_group in range($(8)) {
      r77_left = tl.load($(candidate_vto_copy_40_ptr) + (((((((r77_group % $(8)) * $(2304)) + (r77_row[:, None] * $(48)) + r77_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r77_group % $(8)) * $(2304)) + (r77_row[:, None] * $(48)) + r77_lane[None, :]) / $(48)) % $(48)) * $(48)) + ((((r77_group % $(8)) * $(2304)) + (r77_row[:, None] * $(48)) + r77_lane[None, :]) % $(48))), dtype=tl.bfloat16)
      r77_right = tl.load($(candidate_vbmm_6_ptr) + (((((((r77_group % $(8)) * $(6144)) + (r77_lane[:, None] * $(128)) + r77_col[None, :]) / $(128)) % $(48)) * $(1024)) + ((((((r77_group % $(8)) * $(6144)) + (r77_lane[:, None] * $(128)) + r77_col[None, :]) / $(6144)) % $(8)) * $(128)) + ((((r77_group % $(8)) * $(6144)) + (r77_lane[:, None] * $(128)) + r77_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r77_value = tl.dot(r77_left, r77_right)
      r77_value = r77_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_8_ptr) + ((r77_group * $(6144)) + (r77_row[:, None] * $(128)) + r77_col[None, :]), r77_value)
    }
    r78_z = tl.arange($(384))
    r78_lane = tl.arange($(48))
    r78_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r78_z[:, None] / $(48)) % $(8)) * $(2304)) + ((r78_z[:, None] % $(48)) * $(48)) + (r78_lane[None, :] % $(48))), dtype=tl.float32)
    r78_maximum = tl.max(r78_row_values, axis=1, keep_dims=true)
    r78_weights = tl.exp(r78_row_values - r78_maximum)
    r78_total = tl.sum(r78_weights, axis=1)
    r78_maximum = tl.max(r78_row_values, axis=1)
    r78_value = r78_maximum + tl.log(r78_total)
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(384)])
    tl.store($(candidate_vlogsumexp_3_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(24576))
    r79_value = tl.load($(candidate_vcat_7_ptr) + ((((r79_z / $(192)) % $(16)) * $(1536)) + (((r79_z / $(3072)) % $(8)) * $(192)) + (r79_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(24576)])
    tl.store($(candidate_vto_copy_41_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(24576))
    r80_value = tl.load($(candidate_vcat_8_ptr) + ((((r80_z / $(192)) % $(16)) * $(1536)) + (((r80_z / $(3072)) % $(8)) * $(192)) + (r80_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(24576)])
    tl.store($(candidate_vto_copy_42_ptr) + r80_z, r80_value)
    r81_row = tl.arange($(16))
    r81_col = tl.arange($(16))
    r81_lane = tl.arange($(192))
    for r81_group in range($(8)) {
      r81_left = tl.load($(candidate_vto_copy_41_ptr) + (((((((r81_group % $(8)) * $(3072)) + (r81_row[:, None] * $(192)) + r81_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r81_group % $(8)) * $(3072)) + (r81_row[:, None] * $(192)) + r81_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r81_group % $(8)) * $(3072)) + (r81_row[:, None] * $(192)) + r81_lane[None, :]) % $(192))), dtype=tl.float32)
      r81_right = tl.load($(candidate_vto_copy_42_ptr) + (((((((r81_group % $(8)) * $(3072)) + (r81_lane[:, None] * $(16)) + r81_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r81_group % $(8)) * $(3072)) + (r81_lane[:, None] * $(16)) + r81_col[None, :]) % $(16)) * $(192)) + (((((r81_group % $(8)) * $(3072)) + (r81_lane[:, None] * $(16)) + r81_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r81_value = tl.dot(r81_left, r81_right)
      r81_value = r81_value.to(tl.float32)
      tl.store($(candidate_vbmm_9_ptr) + ((r81_group * $(256)) + (r81_row[:, None] * $(16)) + r81_col[None, :]), r81_value)
    }
    r82_z = tl.arange($(2048))
    r82_value = (tl.load($(candidate_vbmm_9_ptr) + ((((r82_z / $(256)) % $(8)) * $(256)) + (((r82_z / $(16)) % $(16)) * $(16)) + (r82_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(2048)])
    tl.store($(candidate_vmul_25_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(16))
    r83_value = (tl.cast(r83_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r83_value = tl.broadcast_to(r83_value, [$(16)])
    tl.store($(candidate_vadd_13_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(16))
    r84_value = (tl.load($(candidate_vadd_13_ptr) + r84_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r84_value = tl.broadcast_to(r84_value, [$(16)])
    tl.store($(candidate_vsub_10_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(256))
    r85_value = (tl.cast((r85_z % $(16)), tl.int64) > tl.load($(candidate_vsub_10_ptr) + ((r85_z / $(16)) % $(16)), dtype=tl.int64))
    r85_value = tl.broadcast_to(r85_value, [$(256)])
    tl.store($(candidate_vgt_4_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(2048))
    r86_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r86_z / $(16)) % $(16)) * $(16)) + (r86_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_25_ptr) + ((((r86_z / $(256)) % $(8)) * $(256)) + (((r86_z / $(16)) % $(16)) * $(16)) + (r86_z % $(16))), dtype=tl.float32))
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(2048))
    r87_lane = tl.arange($(16))
    r87_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r87_z[:, None] / $(256)) % $(8)) * $(256)) + (((r87_z[:, None] / $(16)) % $(16)) * $(16)) + (r87_lane[None, :] % $(16))), dtype=tl.float32)
    r87_maximum = tl.max(r87_row_values, axis=1, keep_dims=true)
    r87_weights = tl.exp(r87_row_values - r87_maximum)
    r87_total = tl.sum(r87_weights, axis=1)
    r87_maximum = tl.max(r87_row_values, axis=1)
    r87_value = tl.exp(tl.load($(candidate_vmasked_fill_4_ptr) + ((((r87_z / $(256)) % $(8)) * $(256)) + (((r87_z / $(16)) % $(16)) * $(16)) + (r87_z % $(16))), dtype=tl.float32) - r87_maximum) / r87_total
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(2048)])
    tl.store($(candidate_vsoftmax_4_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(2048))
    r88_value = tl.load($(candidate_vsoftmax_4_ptr) + ((((r88_z / $(256)) % $(8)) * $(256)) + (((r88_z / $(16)) % $(16)) * $(16)) + (r88_z % $(16))), dtype=tl.float32)
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(2048)])
    tl.store($(candidate_vto_copy_43_ptr) + r88_z, r88_value)
    r89_row = tl.arange($(16))
    r89_col = tl.arange($(128))
    r89_lane = tl.arange($(16))
    for r89_group in range($(8)) {
      r89_left = tl.load($(candidate_vto_copy_43_ptr) + (((((((r89_group % $(8)) * $(256)) + (r89_row[:, None] * $(16)) + r89_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r89_group % $(8)) * $(256)) + (r89_row[:, None] * $(16)) + r89_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r89_group % $(8)) * $(256)) + (r89_row[:, None] * $(16)) + r89_lane[None, :]) % $(16))), dtype=tl.bfloat16)
      r89_right = tl.load($(candidate_vbmm_6_ptr) + (((((((r89_group % $(8)) * $(2048)) + (r89_lane[:, None] * $(128)) + r89_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r89_group % $(8)) * $(2048)) + (r89_lane[:, None] * $(128)) + r89_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r89_group % $(8)) * $(2048)) + (r89_lane[:, None] * $(128)) + r89_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r89_value = tl.dot(r89_left, r89_right)
      r89_value = r89_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_10_ptr) + ((r89_group * $(2048)) + (r89_row[:, None] * $(128)) + r89_col[None, :]), r89_value)
    }
    r90_z = tl.arange($(128))
    r90_lane = tl.arange($(16))
    r90_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r90_z[:, None] / $(16)) % $(8)) * $(256)) + ((r90_z[:, None] % $(16)) * $(16)) + (r90_lane[None, :] % $(16))), dtype=tl.float32)
    r90_maximum = tl.max(r90_row_values, axis=1, keep_dims=true)
    r90_weights = tl.exp(r90_row_values - r90_maximum)
    r90_total = tl.sum(r90_weights, axis=1)
    r90_maximum = tl.max(r90_row_values, axis=1)
    r90_value = r90_maximum + tl.log(r90_total)
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(128)])
    tl.store($(candidate_vlogsumexp_4_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(98304))
    r91_value = tl.load($(candidate_vcat_7_ptr) + ((((r91_z / $(192)) % $(64)) * $(1536)) + (((r91_z / $(12288)) % $(8)) * $(192)) + (r91_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(98304)])
    tl.store($(candidate_vto_copy_44_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(98304))
    r92_value = tl.load($(candidate_vcat_8_ptr) + ((((r92_z / $(192)) % $(64)) * $(1536)) + (((r92_z / $(12288)) % $(8)) * $(192)) + (r92_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(98304)])
    tl.store($(candidate_vto_copy_45_ptr) + r92_z, r92_value)
    r93_row = tl.arange($(64))
    r93_col = tl.arange($(64))
    r93_lane = tl.arange($(192))
    for r93_group in range($(8)) {
      r93_left = tl.load($(candidate_vto_copy_44_ptr) + (((((((r93_group % $(8)) * $(12288)) + (r93_row[:, None] * $(192)) + r93_lane[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r93_group % $(8)) * $(12288)) + (r93_row[:, None] * $(192)) + r93_lane[None, :]) / $(192)) % $(64)) * $(192)) + ((((r93_group % $(8)) * $(12288)) + (r93_row[:, None] * $(192)) + r93_lane[None, :]) % $(192))), dtype=tl.float32)
      r93_right = tl.load($(candidate_vto_copy_45_ptr) + (((((((r93_group % $(8)) * $(12288)) + (r93_lane[:, None] * $(64)) + r93_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r93_group % $(8)) * $(12288)) + (r93_lane[:, None] * $(64)) + r93_col[None, :]) % $(64)) * $(192)) + (((((r93_group % $(8)) * $(12288)) + (r93_lane[:, None] * $(64)) + r93_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r93_value = tl.dot(r93_left, r93_right)
      r93_value = r93_value.to(tl.float32)
      tl.store($(candidate_vbmm_11_ptr) + ((r93_group * $(4096)) + (r93_row[:, None] * $(64)) + r93_col[None, :]), r93_value)
    }
    r94_z = tl.arange($(32768))
    r94_value = (tl.load($(candidate_vbmm_11_ptr) + ((((r94_z / $(4096)) % $(8)) * $(4096)) + (((r94_z / $(64)) % $(64)) * $(64)) + (r94_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(32768)])
    tl.store($(candidate_vmul_26_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(64))
    r95_value = (tl.cast(r95_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r95_value = tl.broadcast_to(r95_value, [$(64)])
    tl.store($(candidate_vadd_14_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(64))
    r96_value = (tl.load($(candidate_vadd_14_ptr) + r96_z, dtype=tl.int64) - tl.full([], $(64), dtype=tl.int64))
    r96_value = tl.broadcast_to(r96_value, [$(64)])
    tl.store($(candidate_vsub_11_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(4096))
    r97_value = (tl.cast((r97_z % $(64)), tl.int64) > tl.load($(candidate_vsub_11_ptr) + ((r97_z / $(64)) % $(64)), dtype=tl.int64))
    r97_value = tl.broadcast_to(r97_value, [$(4096)])
    tl.store($(candidate_vgt_5_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(32768))
    r98_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r98_z / $(64)) % $(64)) * $(64)) + (r98_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_26_ptr) + ((((r98_z / $(4096)) % $(8)) * $(4096)) + (((r98_z / $(64)) % $(64)) * $(64)) + (r98_z % $(64))), dtype=tl.float32))
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(32768))
    r99_lane = tl.arange($(64))
    r99_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r99_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r99_z[:, None] / $(64)) % $(64)) * $(64)) + (r99_lane[None, :] % $(64))), dtype=tl.float32)
    r99_maximum = tl.max(r99_row_values, axis=1, keep_dims=true)
    r99_weights = tl.exp(r99_row_values - r99_maximum)
    r99_total = tl.sum(r99_weights, axis=1)
    r99_maximum = tl.max(r99_row_values, axis=1)
    r99_value = tl.exp(tl.load($(candidate_vmasked_fill_5_ptr) + ((((r99_z / $(4096)) % $(8)) * $(4096)) + (((r99_z / $(64)) % $(64)) * $(64)) + (r99_z % $(64))), dtype=tl.float32) - r99_maximum) / r99_total
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(32768)])
    tl.store($(candidate_vsoftmax_5_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(32768))
    r100_value = tl.load($(candidate_vsoftmax_5_ptr) + ((((r100_z / $(4096)) % $(8)) * $(4096)) + (((r100_z / $(64)) % $(64)) * $(64)) + (r100_z % $(64))), dtype=tl.float32)
    r100_value = r100_value.to(tl.bfloat16)
    r100_value = tl.broadcast_to(r100_value, [$(32768)])
    tl.store($(candidate_vto_copy_46_ptr) + r100_z, r100_value)
    r101_row = tl.arange($(64))
    r101_col = tl.arange($(128))
    r101_lane = tl.arange($(64))
    for r101_group in range($(8)) {
      r101_left = tl.load($(candidate_vto_copy_46_ptr) + (((((((r101_group % $(8)) * $(4096)) + (r101_row[:, None] * $(64)) + r101_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r101_group % $(8)) * $(4096)) + (r101_row[:, None] * $(64)) + r101_lane[None, :]) / $(64)) % $(64)) * $(64)) + ((((r101_group % $(8)) * $(4096)) + (r101_row[:, None] * $(64)) + r101_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r101_right = tl.load($(candidate_vbmm_6_ptr) + (((((((r101_group % $(8)) * $(8192)) + (r101_lane[:, None] * $(128)) + r101_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r101_group % $(8)) * $(8192)) + (r101_lane[:, None] * $(128)) + r101_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r101_group % $(8)) * $(8192)) + (r101_lane[:, None] * $(128)) + r101_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r101_value = tl.dot(r101_left, r101_right)
      r101_value = r101_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r101_group * $(8192)) + (r101_row[:, None] * $(128)) + r101_col[None, :]), r101_value)
    }
    r102_z = tl.arange($(512))
    r102_lane = tl.arange($(64))
    r102_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r102_z[:, None] / $(64)) % $(8)) * $(4096)) + ((r102_z[:, None] % $(64)) * $(64)) + (r102_lane[None, :] % $(64))), dtype=tl.float32)
    r102_maximum = tl.max(r102_row_values, axis=1, keep_dims=true)
    r102_weights = tl.exp(r102_row_values - r102_maximum)
    r102_total = tl.sum(r102_weights, axis=1)
    r102_maximum = tl.max(r102_row_values, axis=1)
    r102_value = r102_maximum + tl.log(r102_total)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(512)])
    tl.store($(candidate_vlogsumexp_5_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(131072))
    r103_value = tl.where(((r103_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vbmm_8_ptr) + ((((((((r103_z / $(128)) % $(8)) * $(6144)) + ((((r103_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r103_z % $(128))) / $(6144)) % $(8)) * $(6144)) + ((((((r103_z / $(128)) % $(8)) * $(48)) + (((r103_z / $(1024)) % $(128)) - $(0))) % $(48)) * $(128)) + (r103_z % $(128))), dtype=tl.bfloat16), tl.where(((r103_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vbmm_10_ptr) + ((((((((r103_z / $(128)) % $(8)) * $(2048)) + ((((r103_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r103_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r103_z / $(128)) % $(8)) * $(16)) + (((r103_z / $(1024)) % $(128)) - $(48))) % $(16)) * $(128)) + (r103_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_12_ptr) + ((((((((r103_z / $(128)) % $(8)) * $(8192)) + ((((r103_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r103_z % $(128))) / $(8192)) % $(8)) * $(8192)) + ((((((r103_z / $(128)) % $(8)) * $(64)) + (((r103_z / $(1024)) % $(128)) - $(64))) % $(64)) * $(128)) + (r103_z % $(128))), dtype=tl.bfloat16)))
    r103_value = r103_value.to(tl.bfloat16)
    r103_value = tl.broadcast_to(r103_value, [$(131072)])
    tl.store($(candidate_vcat_9_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(1024))
    r104_value = tl.where(((r104_z / $(8)) % $(128)) < $(48), tl.load($(candidate_vlogsumexp_3_ptr) + (((r104_z % $(8)) * $(48)) + ((((r104_z / $(8)) % $(128)) - $(0)) % $(48))), dtype=tl.float32), tl.where(((r104_z / $(8)) % $(128)) < $(64), tl.load($(candidate_vlogsumexp_4_ptr) + (((r104_z % $(8)) * $(16)) + ((((r104_z / $(8)) % $(128)) - $(48)) % $(16))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_5_ptr) + (((r104_z % $(8)) * $(64)) + ((((r104_z / $(8)) % $(128)) - $(64)) % $(64))), dtype=tl.float32)))
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(1024)])
    tl.store($(candidate_vcat_10_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(73728))
    r105_value = tl.load($(candidate_vcat_7_ptr) + ((((r105_z / $(192)) % $(48)) * $(1536)) + (((r105_z / $(9216)) % $(8)) * $(192)) + (r105_z % $(192))), dtype=tl.bfloat16)
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(73728)])
    tl.store($(candidate_vto_copy_47_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(73728))
    r106_value = tl.load($(candidate_vcat_8_ptr) + ((((r106_z / $(192)) % $(48)) * $(1536)) + (((r106_z / $(9216)) % $(8)) * $(192)) + (r106_z % $(192))), dtype=tl.bfloat16)
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(73728)])
    tl.store($(candidate_vto_copy_48_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(49152))
    r107_value = tl.load($(candidate_vbmm_6_ptr) + ((((r107_z / $(128)) % $(48)) * $(1024)) + (((r107_z / $(6144)) % $(8)) * $(128)) + (r107_z % $(128))), dtype=tl.bfloat16)
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(49152)])
    tl.store($(candidate_vto_copy_49_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(49152))
    r108_value = tl.load($(candidate_vmm_10_ptr) + ((((r108_z / $(128)) % $(48)) * $(1024)) + (((r108_z / $(6144)) % $(8)) * $(128)) + (r108_z % $(128))), dtype=tl.bfloat16)
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(49152)])
    tl.store($(candidate_vto_copy_50_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(49152))
    r109_value = tl.load($(candidate_vcat_9_ptr) + ((((r109_z / $(128)) % $(48)) * $(1024)) + (((r109_z / $(6144)) % $(8)) * $(128)) + (r109_z % $(128))), dtype=tl.bfloat16)
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(49152)])
    tl.store($(candidate_vto_copy_51_ptr) + r109_z, r109_value)
    r110_row = tl.arange($(48))
    r110_col = tl.arange($(48))
    r110_lane = tl.arange($(192))
    for r110_group in range($(8)) {
      r110_left = tl.load($(candidate_vto_copy_47_ptr) + (((((((r110_group % $(8)) * $(9216)) + (r110_row[:, None] * $(192)) + r110_lane[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r110_group % $(8)) * $(9216)) + (r110_row[:, None] * $(192)) + r110_lane[None, :]) / $(192)) % $(48)) * $(192)) + ((((r110_group % $(8)) * $(9216)) + (r110_row[:, None] * $(192)) + r110_lane[None, :]) % $(192))), dtype=tl.float32)
      r110_right = tl.load($(candidate_vto_copy_48_ptr) + (((((((r110_group % $(8)) * $(9216)) + (r110_lane[:, None] * $(48)) + r110_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r110_group % $(8)) * $(9216)) + (r110_lane[:, None] * $(48)) + r110_col[None, :]) % $(48)) * $(192)) + (((((r110_group % $(8)) * $(9216)) + (r110_lane[:, None] * $(48)) + r110_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r110_value = tl.dot(r110_left, r110_right)
      r110_value = r110_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r110_group * $(2304)) + (r110_row[:, None] * $(48)) + r110_col[None, :]), r110_value)
    }
    r111_z = tl.arange($(18432))
    r111_value = (tl.load($(candidate_vbmm_13_ptr) + ((((r111_z / $(2304)) % $(8)) * $(2304)) + (((r111_z / $(48)) % $(48)) * $(48)) + (r111_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(18432)])
    tl.store($(candidate_vmul_27_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(48))
    r112_value = (tl.cast(r112_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r112_value = tl.broadcast_to(r112_value, [$(48)])
    tl.store($(candidate_vadd_15_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(48))
    r113_value = (tl.load($(candidate_vadd_15_ptr) + r113_z, dtype=tl.int64) - tl.full([], $(48), dtype=tl.int64))
    r113_value = tl.broadcast_to(r113_value, [$(48)])
    tl.store($(candidate_vsub_12_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(2304))
    r114_value = (tl.cast((r114_z % $(48)), tl.int64) > tl.load($(candidate_vsub_12_ptr) + ((r114_z / $(48)) % $(48)), dtype=tl.int64))
    r114_value = tl.broadcast_to(r114_value, [$(2304)])
    tl.store($(candidate_vgt_6_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(18432))
    r115_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r115_z / $(48)) % $(48)) * $(48)) + (r115_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_27_ptr) + ((((r115_z / $(2304)) % $(8)) * $(2304)) + (((r115_z / $(48)) % $(48)) * $(48)) + (r115_z % $(48))), dtype=tl.float32))
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_6_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(18432))
    r116_value = (tl.load($(candidate_vmasked_fill_6_ptr) + ((((r116_z / $(2304)) % $(8)) * $(2304)) + (((r116_z / $(48)) % $(48)) * $(48)) + (r116_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_10_ptr) + ((((r116_z / $(48)) % $(48)) * $(8)) + ((r116_z / $(2304)) % $(8))), dtype=tl.float32))
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(18432)])
    tl.store($(candidate_vsub_13_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(18432))
    r117_value = tl.exp(tl.load($(candidate_vsub_13_ptr) + ((((r117_z / $(2304)) % $(8)) * $(2304)) + (((r117_z / $(48)) % $(48)) * $(48)) + (r117_z % $(48))), dtype=tl.float32))
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(18432)])
    tl.store($(candidate_vexp_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(49152))
    r118_value = (tl.load($(candidate_vto_copy_50_ptr) + ((((r118_z / $(6144)) % $(8)) * $(6144)) + (((r118_z / $(128)) % $(48)) * $(128)) + (r118_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_51_ptr) + ((((r118_z / $(6144)) % $(8)) * $(6144)) + (((r118_z / $(128)) % $(48)) * $(128)) + (r118_z % $(128))), dtype=tl.float32))
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(49152)])
    tl.store($(candidate_vmul_28_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(384))
    r119_lane = tl.arange($(128))
    r119_value = tl.sum(tl.load($(candidate_vmul_28_ptr) + ((((r119_z[:, None] / $(48)) % $(8)) * $(6144)) + ((r119_z[:, None] % $(48)) * $(128)) + r119_lane[None, :]), dtype=tl.float32), axis=1)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(384)])
    tl.store($(candidate_vsum_1_ptr) + r119_z, r119_value)
    r120_row = tl.arange($(48))
    r120_col = tl.arange($(48))
    r120_lane = tl.arange($(128))
    for r120_group in range($(8)) {
      r120_left = tl.load($(candidate_vto_copy_50_ptr) + (((((((r120_group % $(8)) * $(6144)) + (r120_row[:, None] * $(128)) + r120_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r120_group % $(8)) * $(6144)) + (r120_row[:, None] * $(128)) + r120_lane[None, :]) / $(128)) % $(48)) * $(128)) + ((((r120_group % $(8)) * $(6144)) + (r120_row[:, None] * $(128)) + r120_lane[None, :]) % $(128))), dtype=tl.float32)
      r120_right = tl.load($(candidate_vto_copy_49_ptr) + (((((((r120_group % $(8)) * $(6144)) + (r120_lane[:, None] * $(48)) + r120_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r120_group % $(8)) * $(6144)) + (r120_lane[:, None] * $(48)) + r120_col[None, :]) % $(48)) * $(128)) + (((((r120_group % $(8)) * $(6144)) + (r120_lane[:, None] * $(48)) + r120_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r120_value = tl.dot(r120_left, r120_right)
      r120_value = r120_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r120_group * $(2304)) + (r120_row[:, None] * $(48)) + r120_col[None, :]), r120_value)
    }
    r121_z = tl.arange($(18432))
    r121_value = (tl.load($(candidate_vbmm_14_ptr) + ((((r121_z / $(2304)) % $(8)) * $(2304)) + (((r121_z / $(48)) % $(48)) * $(48)) + (r121_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_1_ptr) + ((((r121_z / $(2304)) % $(8)) * $(48)) + ((r121_z / $(48)) % $(48))), dtype=tl.float32))
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(18432)])
    tl.store($(candidate_vsub_14_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(18432))
    r122_value = (tl.load($(candidate_vexp_ptr) + ((((r122_z / $(2304)) % $(8)) * $(2304)) + (((r122_z / $(48)) % $(48)) * $(48)) + (r122_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_14_ptr) + ((((r122_z / $(2304)) % $(8)) * $(2304)) + (((r122_z / $(48)) % $(48)) * $(48)) + (r122_z % $(48))), dtype=tl.float32))
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(18432)])
    tl.store($(candidate_vmul_29_ptr) + r122_z, r122_value)
    r123_row = tl.arange($(48))
    r123_col = tl.arange($(192))
    r123_lane = tl.arange($(48))
    for r123_group in range($(8)) {
      r123_left = tl.load($(candidate_vmul_29_ptr) + (((((((r123_group % $(8)) * $(2304)) + (r123_row[:, None] * $(48)) + r123_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r123_group % $(8)) * $(2304)) + (r123_row[:, None] * $(48)) + r123_lane[None, :]) % $(48)) * $(48)) + (((((r123_group % $(8)) * $(2304)) + (r123_row[:, None] * $(48)) + r123_lane[None, :]) / $(48)) % $(48))), dtype=tl.float32)
      r123_right = tl.load($(candidate_vto_copy_47_ptr) + (((((((r123_group % $(8)) * $(9216)) + (r123_lane[:, None] * $(192)) + r123_col[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r123_group % $(8)) * $(9216)) + (r123_lane[:, None] * $(192)) + r123_col[None, :]) / $(192)) % $(48)) * $(192)) + ((((r123_group % $(8)) * $(9216)) + (r123_lane[:, None] * $(192)) + r123_col[None, :]) % $(192))), dtype=tl.float32)
      r123_value = tl.dot(r123_left, r123_right)
      r123_value = r123_value.to(tl.float32)
      tl.store($(candidate_vbmm_16_ptr) + ((r123_group * $(9216)) + (r123_row[:, None] * $(192)) + r123_col[None, :]), r123_value)
    }
    r124_z = tl.arange($(73728))
    r124_value = (tl.load($(candidate_vbmm_16_ptr) + ((((r124_z / $(9216)) % $(8)) * $(9216)) + (((r124_z / $(192)) % $(48)) * $(192)) + (r124_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(73728)])
    tl.store($(candidate_vmul_31_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(73728))
    r125_value = tl.load($(candidate_vmul_31_ptr) + ((((r125_z / $(192)) % $(8)) * $(9216)) + (((r125_z / $(1536)) % $(48)) * $(192)) + (r125_z % $(192))), dtype=tl.float32)
    r125_value = r125_value.to(tl.bfloat16)
    r125_value = tl.broadcast_to(r125_value, [$(73728)])
    tl.store($(candidate_vto_copy_53_ptr) + r125_z, r125_value)
    r126_row = tl.arange($(48))
    r126_col = tl.arange($(128))
    r126_lane = tl.arange($(48))
    for r126_group in range($(8)) {
      r126_left = tl.load($(candidate_vexp_ptr) + (((((((r126_group % $(8)) * $(2304)) + (r126_row[:, None] * $(48)) + r126_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r126_group % $(8)) * $(2304)) + (r126_row[:, None] * $(48)) + r126_lane[None, :]) % $(48)) * $(48)) + (((((r126_group % $(8)) * $(2304)) + (r126_row[:, None] * $(48)) + r126_lane[None, :]) / $(48)) % $(48))), dtype=tl.float32)
      r126_right = tl.load($(candidate_vto_copy_50_ptr) + (((((((r126_group % $(8)) * $(6144)) + (r126_lane[:, None] * $(128)) + r126_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r126_group % $(8)) * $(6144)) + (r126_lane[:, None] * $(128)) + r126_col[None, :]) / $(128)) % $(48)) * $(128)) + ((((r126_group % $(8)) * $(6144)) + (r126_lane[:, None] * $(128)) + r126_col[None, :]) % $(128))), dtype=tl.float32)
      r126_value = tl.dot(r126_left, r126_right)
      r126_value = r126_value.to(tl.float32)
      tl.store($(candidate_vbmm_17_ptr) + ((r126_group * $(6144)) + (r126_row[:, None] * $(128)) + r126_col[None, :]), r126_value)
    }
    r127_z = tl.arange($(49152))
    r127_value = tl.load($(candidate_vbmm_17_ptr) + ((((r127_z / $(128)) % $(8)) * $(6144)) + (((r127_z / $(1024)) % $(48)) * $(128)) + (r127_z % $(128))), dtype=tl.float32)
    r127_value = r127_value.to(tl.bfloat16)
    r127_value = tl.broadcast_to(r127_value, [$(49152)])
    tl.store($(candidate_vto_copy_54_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(24576))
    r128_value = tl.load($(candidate_vcat_7_ptr) + ((((r128_z / $(192)) % $(16)) * $(1536)) + (((r128_z / $(3072)) % $(8)) * $(192)) + (r128_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(24576)])
    tl.store($(candidate_vto_copy_55_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(24576))
    r129_value = tl.load($(candidate_vcat_8_ptr) + ((((r129_z / $(192)) % $(16)) * $(1536)) + (((r129_z / $(3072)) % $(8)) * $(192)) + (r129_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(24576)])
    tl.store($(candidate_vto_copy_56_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(16384))
    r130_value = tl.load($(candidate_vbmm_6_ptr) + ((((r130_z / $(128)) % $(16)) * $(1024)) + (((r130_z / $(2048)) % $(8)) * $(128)) + (r130_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(16384)])
    tl.store($(candidate_vto_copy_57_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(16384))
    r131_value = tl.load($(candidate_vmm_10_ptr) + ((((r131_z / $(128)) % $(16)) * $(1024)) + (((r131_z / $(2048)) % $(8)) * $(128)) + (r131_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(16384)])
    tl.store($(candidate_vto_copy_58_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(16384))
    r132_value = tl.load($(candidate_vcat_9_ptr) + ((((r132_z / $(128)) % $(16)) * $(1024)) + (((r132_z / $(2048)) % $(8)) * $(128)) + (r132_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(16384)])
    tl.store($(candidate_vto_copy_59_ptr) + r132_z, r132_value)
    r133_row = tl.arange($(16))
    r133_col = tl.arange($(16))
    r133_lane = tl.arange($(192))
    for r133_group in range($(8)) {
      r133_left = tl.load($(candidate_vto_copy_55_ptr) + (((((((r133_group % $(8)) * $(3072)) + (r133_row[:, None] * $(192)) + r133_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r133_group % $(8)) * $(3072)) + (r133_row[:, None] * $(192)) + r133_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r133_group % $(8)) * $(3072)) + (r133_row[:, None] * $(192)) + r133_lane[None, :]) % $(192))), dtype=tl.float32)
      r133_right = tl.load($(candidate_vto_copy_56_ptr) + (((((((r133_group % $(8)) * $(3072)) + (r133_lane[:, None] * $(16)) + r133_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r133_group % $(8)) * $(3072)) + (r133_lane[:, None] * $(16)) + r133_col[None, :]) % $(16)) * $(192)) + (((((r133_group % $(8)) * $(3072)) + (r133_lane[:, None] * $(16)) + r133_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r133_value = tl.dot(r133_left, r133_right)
      r133_value = r133_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r133_group * $(256)) + (r133_row[:, None] * $(16)) + r133_col[None, :]), r133_value)
    }
    r134_z = tl.arange($(2048))
    r134_value = (tl.load($(candidate_vbmm_18_ptr) + ((((r134_z / $(256)) % $(8)) * $(256)) + (((r134_z / $(16)) % $(16)) * $(16)) + (r134_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(2048)])
    tl.store($(candidate_vmul_32_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(16))
    r135_value = (tl.cast(r135_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r135_value = tl.broadcast_to(r135_value, [$(16)])
    tl.store($(candidate_vadd_16_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(16))
    r136_value = (tl.load($(candidate_vadd_16_ptr) + r136_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r136_value = tl.broadcast_to(r136_value, [$(16)])
    tl.store($(candidate_vsub_15_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(256))
    r137_value = (tl.cast((r137_z % $(16)), tl.int64) > tl.load($(candidate_vsub_15_ptr) + ((r137_z / $(16)) % $(16)), dtype=tl.int64))
    r137_value = tl.broadcast_to(r137_value, [$(256)])
    tl.store($(candidate_vgt_7_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(2048))
    r138_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r138_z / $(16)) % $(16)) * $(16)) + (r138_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_32_ptr) + ((((r138_z / $(256)) % $(8)) * $(256)) + (((r138_z / $(16)) % $(16)) * $(16)) + (r138_z % $(16))), dtype=tl.float32))
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_7_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(2048))
    r139_value = (tl.load($(candidate_vmasked_fill_7_ptr) + ((((r139_z / $(256)) % $(8)) * $(256)) + (((r139_z / $(16)) % $(16)) * $(16)) + (r139_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_10_ptr) + ((((r139_z / $(16)) % $(16)) * $(8)) + ((r139_z / $(256)) % $(8)) + $(384)), dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(2048)])
    tl.store($(candidate_vsub_16_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(2048))
    r140_value = tl.exp(tl.load($(candidate_vsub_16_ptr) + ((((r140_z / $(256)) % $(8)) * $(256)) + (((r140_z / $(16)) % $(16)) * $(16)) + (r140_z % $(16))), dtype=tl.float32))
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(2048)])
    tl.store($(candidate_vexp_1_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(16384))
    r141_value = (tl.load($(candidate_vto_copy_58_ptr) + ((((r141_z / $(2048)) % $(8)) * $(2048)) + (((r141_z / $(128)) % $(16)) * $(128)) + (r141_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_59_ptr) + ((((r141_z / $(2048)) % $(8)) * $(2048)) + (((r141_z / $(128)) % $(16)) * $(128)) + (r141_z % $(128))), dtype=tl.float32))
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(16384)])
    tl.store($(candidate_vmul_33_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(128))
    r142_lane = tl.arange($(128))
    r142_value = tl.sum(tl.load($(candidate_vmul_33_ptr) + ((((r142_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r142_z[:, None] % $(16)) * $(128)) + r142_lane[None, :]), dtype=tl.float32), axis=1)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(128)])
    tl.store($(candidate_vsum_2_ptr) + r142_z, r142_value)
    r143_row = tl.arange($(16))
    r143_col = tl.arange($(16))
    r143_lane = tl.arange($(128))
    for r143_group in range($(8)) {
      r143_left = tl.load($(candidate_vto_copy_58_ptr) + (((((((r143_group % $(8)) * $(2048)) + (r143_row[:, None] * $(128)) + r143_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r143_group % $(8)) * $(2048)) + (r143_row[:, None] * $(128)) + r143_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r143_group % $(8)) * $(2048)) + (r143_row[:, None] * $(128)) + r143_lane[None, :]) % $(128))), dtype=tl.float32)
      r143_right = tl.load($(candidate_vto_copy_57_ptr) + (((((((r143_group % $(8)) * $(2048)) + (r143_lane[:, None] * $(16)) + r143_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r143_group % $(8)) * $(2048)) + (r143_lane[:, None] * $(16)) + r143_col[None, :]) % $(16)) * $(128)) + (((((r143_group % $(8)) * $(2048)) + (r143_lane[:, None] * $(16)) + r143_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r143_value = tl.dot(r143_left, r143_right)
      r143_value = r143_value.to(tl.float32)
      tl.store($(candidate_vbmm_19_ptr) + ((r143_group * $(256)) + (r143_row[:, None] * $(16)) + r143_col[None, :]), r143_value)
    }
    r144_z = tl.arange($(2048))
    r144_value = (tl.load($(candidate_vbmm_19_ptr) + ((((r144_z / $(256)) % $(8)) * $(256)) + (((r144_z / $(16)) % $(16)) * $(16)) + (r144_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_2_ptr) + ((((r144_z / $(256)) % $(8)) * $(16)) + ((r144_z / $(16)) % $(16))), dtype=tl.float32))
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(2048)])
    tl.store($(candidate_vsub_17_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(2048))
    r145_value = (tl.load($(candidate_vexp_1_ptr) + ((((r145_z / $(256)) % $(8)) * $(256)) + (((r145_z / $(16)) % $(16)) * $(16)) + (r145_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_17_ptr) + ((((r145_z / $(256)) % $(8)) * $(256)) + (((r145_z / $(16)) % $(16)) * $(16)) + (r145_z % $(16))), dtype=tl.float32))
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(2048)])
    tl.store($(candidate_vmul_34_ptr) + r145_z, r145_value)
    r146_row = tl.arange($(16))
    r146_col = tl.arange($(192))
    r146_lane = tl.arange($(16))
    for r146_group in range($(8)) {
      r146_left = tl.load($(candidate_vmul_34_ptr) + (((((((r146_group % $(8)) * $(256)) + (r146_row[:, None] * $(16)) + r146_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r146_group % $(8)) * $(256)) + (r146_row[:, None] * $(16)) + r146_lane[None, :]) % $(16)) * $(16)) + (((((r146_group % $(8)) * $(256)) + (r146_row[:, None] * $(16)) + r146_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r146_right = tl.load($(candidate_vto_copy_55_ptr) + (((((((r146_group % $(8)) * $(3072)) + (r146_lane[:, None] * $(192)) + r146_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r146_group % $(8)) * $(3072)) + (r146_lane[:, None] * $(192)) + r146_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r146_group % $(8)) * $(3072)) + (r146_lane[:, None] * $(192)) + r146_col[None, :]) % $(192))), dtype=tl.float32)
      r146_value = tl.dot(r146_left, r146_right)
      r146_value = r146_value.to(tl.float32)
      tl.store($(candidate_vbmm_21_ptr) + ((r146_group * $(3072)) + (r146_row[:, None] * $(192)) + r146_col[None, :]), r146_value)
    }
    r147_z = tl.arange($(24576))
    r147_value = (tl.load($(candidate_vbmm_21_ptr) + ((((r147_z / $(3072)) % $(8)) * $(3072)) + (((r147_z / $(192)) % $(16)) * $(192)) + (r147_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r147_value = r147_value.to(tl.float32)
    r147_value = tl.broadcast_to(r147_value, [$(24576)])
    tl.store($(candidate_vmul_36_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(24576))
    r148_value = tl.load($(candidate_vmul_36_ptr) + ((((r148_z / $(192)) % $(8)) * $(3072)) + (((r148_z / $(1536)) % $(16)) * $(192)) + (r148_z % $(192))), dtype=tl.float32)
    r148_value = r148_value.to(tl.bfloat16)
    r148_value = tl.broadcast_to(r148_value, [$(24576)])
    tl.store($(candidate_vto_copy_61_ptr) + r148_z, r148_value)
    r149_row = tl.arange($(16))
    r149_col = tl.arange($(128))
    r149_lane = tl.arange($(16))
    for r149_group in range($(8)) {
      r149_left = tl.load($(candidate_vexp_1_ptr) + (((((((r149_group % $(8)) * $(256)) + (r149_row[:, None] * $(16)) + r149_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r149_group % $(8)) * $(256)) + (r149_row[:, None] * $(16)) + r149_lane[None, :]) % $(16)) * $(16)) + (((((r149_group % $(8)) * $(256)) + (r149_row[:, None] * $(16)) + r149_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r149_right = tl.load($(candidate_vto_copy_58_ptr) + (((((((r149_group % $(8)) * $(2048)) + (r149_lane[:, None] * $(128)) + r149_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r149_group % $(8)) * $(2048)) + (r149_lane[:, None] * $(128)) + r149_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r149_group % $(8)) * $(2048)) + (r149_lane[:, None] * $(128)) + r149_col[None, :]) % $(128))), dtype=tl.float32)
      r149_value = tl.dot(r149_left, r149_right)
      r149_value = r149_value.to(tl.float32)
      tl.store($(candidate_vbmm_22_ptr) + ((r149_group * $(2048)) + (r149_row[:, None] * $(128)) + r149_col[None, :]), r149_value)
    }
    r150_z = tl.arange($(16384))
    r150_value = tl.load($(candidate_vbmm_22_ptr) + ((((r150_z / $(128)) % $(8)) * $(2048)) + (((r150_z / $(1024)) % $(16)) * $(128)) + (r150_z % $(128))), dtype=tl.float32)
    r150_value = r150_value.to(tl.bfloat16)
    r150_value = tl.broadcast_to(r150_value, [$(16384)])
    tl.store($(candidate_vto_copy_62_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(98304))
    r151_value = tl.load($(candidate_vcat_7_ptr) + ((((r151_z / $(192)) % $(64)) * $(1536)) + (((r151_z / $(12288)) % $(8)) * $(192)) + (r151_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(98304)])
    tl.store($(candidate_vto_copy_63_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(98304))
    r152_value = tl.load($(candidate_vcat_8_ptr) + ((((r152_z / $(192)) % $(64)) * $(1536)) + (((r152_z / $(12288)) % $(8)) * $(192)) + (r152_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(98304)])
    tl.store($(candidate_vto_copy_64_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(65536))
    r153_value = tl.load($(candidate_vbmm_6_ptr) + ((((r153_z / $(128)) % $(64)) * $(1024)) + (((r153_z / $(8192)) % $(8)) * $(128)) + (r153_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r153_value = r153_value.to(tl.float32)
    r153_value = tl.broadcast_to(r153_value, [$(65536)])
    tl.store($(candidate_vto_copy_65_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(65536))
    r154_value = tl.load($(candidate_vmm_10_ptr) + ((((r154_z / $(128)) % $(64)) * $(1024)) + (((r154_z / $(8192)) % $(8)) * $(128)) + (r154_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(65536)])
    tl.store($(candidate_vto_copy_66_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(65536))
    r155_value = tl.load($(candidate_vcat_9_ptr) + ((((r155_z / $(128)) % $(64)) * $(1024)) + (((r155_z / $(8192)) % $(8)) * $(128)) + (r155_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(65536)])
    tl.store($(candidate_vto_copy_67_ptr) + r155_z, r155_value)
    r156_row = tl.arange($(64))
    r156_col = tl.arange($(64))
    r156_lane = tl.arange($(192))
    for r156_group in range($(8)) {
      r156_left = tl.load($(candidate_vto_copy_63_ptr) + (((((((r156_group % $(8)) * $(12288)) + (r156_row[:, None] * $(192)) + r156_lane[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r156_group % $(8)) * $(12288)) + (r156_row[:, None] * $(192)) + r156_lane[None, :]) / $(192)) % $(64)) * $(192)) + ((((r156_group % $(8)) * $(12288)) + (r156_row[:, None] * $(192)) + r156_lane[None, :]) % $(192))), dtype=tl.float32)
      r156_right = tl.load($(candidate_vto_copy_64_ptr) + (((((((r156_group % $(8)) * $(12288)) + (r156_lane[:, None] * $(64)) + r156_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r156_group % $(8)) * $(12288)) + (r156_lane[:, None] * $(64)) + r156_col[None, :]) % $(64)) * $(192)) + (((((r156_group % $(8)) * $(12288)) + (r156_lane[:, None] * $(64)) + r156_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r156_value = tl.dot(r156_left, r156_right)
      r156_value = r156_value.to(tl.float32)
      tl.store($(candidate_vbmm_23_ptr) + ((r156_group * $(4096)) + (r156_row[:, None] * $(64)) + r156_col[None, :]), r156_value)
    }
    r157_z = tl.arange($(32768))
    r157_value = (tl.load($(candidate_vbmm_23_ptr) + ((((r157_z / $(4096)) % $(8)) * $(4096)) + (((r157_z / $(64)) % $(64)) * $(64)) + (r157_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(32768)])
    tl.store($(candidate_vmul_37_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(64))
    r158_value = (tl.cast(r158_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r158_value = tl.broadcast_to(r158_value, [$(64)])
    tl.store($(candidate_vadd_17_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(64))
    r159_value = (tl.load($(candidate_vadd_17_ptr) + r159_z, dtype=tl.int64) - tl.full([], $(64), dtype=tl.int64))
    r159_value = tl.broadcast_to(r159_value, [$(64)])
    tl.store($(candidate_vsub_18_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(4096))
    r160_value = (tl.cast((r160_z % $(64)), tl.int64) > tl.load($(candidate_vsub_18_ptr) + ((r160_z / $(64)) % $(64)), dtype=tl.int64))
    r160_value = tl.broadcast_to(r160_value, [$(4096)])
    tl.store($(candidate_vgt_8_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(32768))
    r161_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r161_z / $(64)) % $(64)) * $(64)) + (r161_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_37_ptr) + ((((r161_z / $(4096)) % $(8)) * $(4096)) + (((r161_z / $(64)) % $(64)) * $(64)) + (r161_z % $(64))), dtype=tl.float32))
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_8_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(32768))
    r162_value = (tl.load($(candidate_vmasked_fill_8_ptr) + ((((r162_z / $(4096)) % $(8)) * $(4096)) + (((r162_z / $(64)) % $(64)) * $(64)) + (r162_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_10_ptr) + ((((r162_z / $(64)) % $(64)) * $(8)) + ((r162_z / $(4096)) % $(8)) + $(512)), dtype=tl.float32))
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(32768)])
    tl.store($(candidate_vsub_19_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(32768))
    r163_value = tl.exp(tl.load($(candidate_vsub_19_ptr) + ((((r163_z / $(4096)) % $(8)) * $(4096)) + (((r163_z / $(64)) % $(64)) * $(64)) + (r163_z % $(64))), dtype=tl.float32))
    r163_value = r163_value.to(tl.float32)
    r163_value = tl.broadcast_to(r163_value, [$(32768)])
    tl.store($(candidate_vexp_2_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(65536))
    r164_value = (tl.load($(candidate_vto_copy_66_ptr) + ((((r164_z / $(8192)) % $(8)) * $(8192)) + (((r164_z / $(128)) % $(64)) * $(128)) + (r164_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_67_ptr) + ((((r164_z / $(8192)) % $(8)) * $(8192)) + (((r164_z / $(128)) % $(64)) * $(128)) + (r164_z % $(128))), dtype=tl.float32))
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(65536)])
    tl.store($(candidate_vmul_38_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(512))
    r165_lane = tl.arange($(128))
    r165_value = tl.sum(tl.load($(candidate_vmul_38_ptr) + ((((r165_z[:, None] / $(64)) % $(8)) * $(8192)) + ((r165_z[:, None] % $(64)) * $(128)) + r165_lane[None, :]), dtype=tl.float32), axis=1)
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(512)])
    tl.store($(candidate_vsum_3_ptr) + r165_z, r165_value)
    r166_row = tl.arange($(64))
    r166_col = tl.arange($(64))
    r166_lane = tl.arange($(128))
    for r166_group in range($(8)) {
      r166_left = tl.load($(candidate_vto_copy_66_ptr) + (((((((r166_group % $(8)) * $(8192)) + (r166_row[:, None] * $(128)) + r166_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r166_group % $(8)) * $(8192)) + (r166_row[:, None] * $(128)) + r166_lane[None, :]) / $(128)) % $(64)) * $(128)) + ((((r166_group % $(8)) * $(8192)) + (r166_row[:, None] * $(128)) + r166_lane[None, :]) % $(128))), dtype=tl.float32)
      r166_right = tl.load($(candidate_vto_copy_65_ptr) + (((((((r166_group % $(8)) * $(8192)) + (r166_lane[:, None] * $(64)) + r166_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r166_group % $(8)) * $(8192)) + (r166_lane[:, None] * $(64)) + r166_col[None, :]) % $(64)) * $(128)) + (((((r166_group % $(8)) * $(8192)) + (r166_lane[:, None] * $(64)) + r166_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r166_value = tl.dot(r166_left, r166_right)
      r166_value = r166_value.to(tl.float32)
      tl.store($(candidate_vbmm_24_ptr) + ((r166_group * $(4096)) + (r166_row[:, None] * $(64)) + r166_col[None, :]), r166_value)
    }
    r167_z = tl.arange($(32768))
    r167_value = (tl.load($(candidate_vbmm_24_ptr) + ((((r167_z / $(4096)) % $(8)) * $(4096)) + (((r167_z / $(64)) % $(64)) * $(64)) + (r167_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_3_ptr) + ((((r167_z / $(4096)) % $(8)) * $(64)) + ((r167_z / $(64)) % $(64))), dtype=tl.float32))
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(32768)])
    tl.store($(candidate_vsub_20_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(32768))
    r168_value = (tl.load($(candidate_vexp_2_ptr) + ((((r168_z / $(4096)) % $(8)) * $(4096)) + (((r168_z / $(64)) % $(64)) * $(64)) + (r168_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_20_ptr) + ((((r168_z / $(4096)) % $(8)) * $(4096)) + (((r168_z / $(64)) % $(64)) * $(64)) + (r168_z % $(64))), dtype=tl.float32))
    r168_value = r168_value.to(tl.float32)
    r168_value = tl.broadcast_to(r168_value, [$(32768)])
    tl.store($(candidate_vmul_39_ptr) + r168_z, r168_value)
    r169_row = tl.arange($(64))
    r169_col = tl.arange($(192))
    r169_lane = tl.arange($(64))
    for r169_group in range($(8)) {
      r169_left = tl.load($(candidate_vmul_39_ptr) + (((((((r169_group % $(8)) * $(4096)) + (r169_row[:, None] * $(64)) + r169_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r169_group % $(8)) * $(4096)) + (r169_row[:, None] * $(64)) + r169_lane[None, :]) % $(64)) * $(64)) + (((((r169_group % $(8)) * $(4096)) + (r169_row[:, None] * $(64)) + r169_lane[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r169_right = tl.load($(candidate_vto_copy_63_ptr) + (((((((r169_group % $(8)) * $(12288)) + (r169_lane[:, None] * $(192)) + r169_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r169_group % $(8)) * $(12288)) + (r169_lane[:, None] * $(192)) + r169_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r169_group % $(8)) * $(12288)) + (r169_lane[:, None] * $(192)) + r169_col[None, :]) % $(192))), dtype=tl.float32)
      r169_value = tl.dot(r169_left, r169_right)
      r169_value = r169_value.to(tl.float32)
      tl.store($(candidate_vbmm_26_ptr) + ((r169_group * $(12288)) + (r169_row[:, None] * $(192)) + r169_col[None, :]), r169_value)
    }
    r170_z = tl.arange($(98304))
    r170_value = (tl.load($(candidate_vbmm_26_ptr) + ((((r170_z / $(12288)) % $(8)) * $(12288)) + (((r170_z / $(192)) % $(64)) * $(192)) + (r170_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r170_value = r170_value.to(tl.float32)
    r170_value = tl.broadcast_to(r170_value, [$(98304)])
    tl.store($(candidate_vmul_41_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(98304))
    r171_value = tl.load($(candidate_vmul_41_ptr) + ((((r171_z / $(192)) % $(8)) * $(12288)) + (((r171_z / $(1536)) % $(64)) * $(192)) + (r171_z % $(192))), dtype=tl.float32)
    r171_value = r171_value.to(tl.bfloat16)
    r171_value = tl.broadcast_to(r171_value, [$(98304)])
    tl.store($(candidate_vto_copy_69_ptr) + r171_z, r171_value)
    r172_row = tl.arange($(64))
    r172_col = tl.arange($(128))
    r172_lane = tl.arange($(64))
    for r172_group in range($(8)) {
      r172_left = tl.load($(candidate_vexp_2_ptr) + (((((((r172_group % $(8)) * $(4096)) + (r172_row[:, None] * $(64)) + r172_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r172_group % $(8)) * $(4096)) + (r172_row[:, None] * $(64)) + r172_lane[None, :]) % $(64)) * $(64)) + (((((r172_group % $(8)) * $(4096)) + (r172_row[:, None] * $(64)) + r172_lane[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r172_right = tl.load($(candidate_vto_copy_66_ptr) + (((((((r172_group % $(8)) * $(8192)) + (r172_lane[:, None] * $(128)) + r172_col[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r172_group % $(8)) * $(8192)) + (r172_lane[:, None] * $(128)) + r172_col[None, :]) / $(128)) % $(64)) * $(128)) + ((((r172_group % $(8)) * $(8192)) + (r172_lane[:, None] * $(128)) + r172_col[None, :]) % $(128))), dtype=tl.float32)
      r172_value = tl.dot(r172_left, r172_right)
      r172_value = r172_value.to(tl.float32)
      tl.store($(candidate_vbmm_27_ptr) + ((r172_group * $(8192)) + (r172_row[:, None] * $(128)) + r172_col[None, :]), r172_value)
    }
    r173_z = tl.arange($(65536))
    r173_value = tl.load($(candidate_vbmm_27_ptr) + ((((r173_z / $(128)) % $(8)) * $(8192)) + (((r173_z / $(1024)) % $(64)) * $(128)) + (r173_z % $(128))), dtype=tl.float32)
    r173_value = r173_value.to(tl.bfloat16)
    r173_value = tl.broadcast_to(r173_value, [$(65536)])
    tl.store($(candidate_vto_copy_70_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(196608))
    r174_value = tl.where(((r174_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vto_copy_53_ptr) + ((((((r174_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r174_z / $(192)) % $(8)) * $(192)) + (r174_z % $(192))), dtype=tl.bfloat16), tl.where(((r174_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vto_copy_61_ptr) + ((((((r174_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r174_z / $(192)) % $(8)) * $(192)) + (r174_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_69_ptr) + ((((((r174_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r174_z / $(192)) % $(8)) * $(192)) + (r174_z % $(192))), dtype=tl.bfloat16)))
    r174_value = r174_value.to(tl.bfloat16)
    r174_value = tl.broadcast_to(r174_value, [$(196608)])
    tl.store($(candidate_vcat_12_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(131072))
    r175_value = tl.where(((r175_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vto_copy_54_ptr) + ((((((r175_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r175_z / $(128)) % $(8)) * $(128)) + (r175_z % $(128))), dtype=tl.bfloat16), tl.where(((r175_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vto_copy_62_ptr) + ((((((r175_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r175_z / $(128)) % $(8)) * $(128)) + (r175_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_70_ptr) + ((((((r175_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r175_z / $(128)) % $(8)) * $(128)) + (r175_z % $(128))), dtype=tl.bfloat16)))
    r175_value = r175_value.to(tl.bfloat16)
    r175_value = tl.broadcast_to(r175_value, [$(131072)])
    tl.store($(candidate_vcat_13_ptr) + r175_z, r175_value)
    r176_row = tl.arange($(1024))
    r176_col = tl.arange($(512))
    r176_lane = tl.arange($(128))
    for r176_group in range($(1)) {
      r176_left = tl.load($(candidate_vcat_12_ptr) + (((((r176_row[:, None] * $(128)) + r176_lane[None, :]) % $(128)) * $(1536)) + (((((r176_row[:, None] * $(128)) + r176_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r176_row[:, None] * $(128)) + r176_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r176_right = tl.load($(candidate_vto_copy_22_ptr) + ((((((r176_lane[:, None] * $(512)) + r176_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r176_lane[:, None] * $(512)) + r176_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r176_value = tl.dot(r176_left, r176_right)
      r176_value = r176_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_30_ptr) + ((r176_group * $(524288)) + (r176_row[:, None] * $(512)) + r176_col[None, :]), r176_value)
    }
    r177_row = tl.arange($(1024))
    r177_col = tl.arange($(512))
    r177_lane = tl.arange($(128))
    for r177_group in range($(1)) {
      r177_left = tl.load($(candidate_vcat_13_ptr) + (((((r177_row[:, None] * $(128)) + r177_lane[None, :]) % $(128)) * $(1024)) + (((((r177_row[:, None] * $(128)) + r177_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r177_row[:, None] * $(128)) + r177_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r177_right = tl.load($(candidate_vto_copy_22_ptr) + ((((((r177_lane[:, None] * $(512)) + r177_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r177_lane[:, None] * $(512)) + r177_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r177_value = tl.dot(r177_left, r177_right)
      r177_value = r177_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_31_ptr) + ((r177_group * $(524288)) + (r177_row[:, None] * $(512)) + r177_col[None, :]), r177_value)
    }
    r178_z = tl.arange($(1048576))
    r178_value = tl.where(((r178_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_30_ptr) + ((((((((r178_z / $(131072)) % $(8)) * $(65536)) + ((((r178_z / $(512)) % $(256)) - $(0)) * $(512)) + (r178_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r178_z / $(131072)) % $(8)) * $(128)) + (((r178_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r178_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_31_ptr) + ((((((((r178_z / $(131072)) % $(8)) * $(65536)) + ((((r178_z / $(512)) % $(256)) - $(128)) * $(512)) + (r178_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r178_z / $(131072)) % $(8)) * $(128)) + (((r178_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r178_z % $(512))), dtype=tl.bfloat16))
    r178_value = r178_value.to(tl.bfloat16)
    r178_value = tl.broadcast_to(r178_value, [$(1048576)])
    tl.store($(candidate_vcat_14_ptr) + r178_z, r178_value)
    z = tl.arange($(1048576))
    value = tl.load($(candidate_vcat_14_ptr) + ((((((((z / $(512)) % $(2048)) * $(512)) + (z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((z / $(512)) % $(2048)) % $(256)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(1048576)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_varlen_wgrad
