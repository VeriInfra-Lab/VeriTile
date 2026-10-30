import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_nonabsorb_varlen_wgrad
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
    r3_z = tl.arange($(524288))
    r3_value = (tl.load($(parameter2_ptr) + ((((r3_z / $(256)) % $(2048)) * $(256)) + (r3_z % $(256))), dtype=tl.float32) * 0.0625)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(524288)])
    tl.store($(vmul_2_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(163840))
    r4_value = (tl.load($(parameter3_ptr) + ((((r4_z / $(512)) % $(320)) * $(512)) + (r4_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(163840)])
    tl.store($(vmul_3_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(256))
    r5_value = (tl.load($(parameter4_ptr) + r5_z, dtype=tl.float32) * 0.1)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(256)])
    tl.store($(vmul_4_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(256))
    r6_value = (tl.load($(vmul_4_ptr) + r6_z, dtype=tl.float32) + 1.0)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(256)])
    tl.store($(vadd_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(917504))
    r7_value = (tl.load($(parameter5_ptr) + ((((r7_z / $(256)) % $(3584)) * $(256)) + (r7_z % $(256))), dtype=tl.float32) * 0.0625)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(917504)])
    tl.store($(vmul_5_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(1048576))
    r8_value = (tl.load($(parameter6_ptr) + ((((r8_z / $(2048)) % $(512)) * $(2048)) + (r8_z % $(2048))), dtype=tl.float32) * 0.02209708691207961)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(1048576)])
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
    r11_z = tl.arange($(524288))
    r11_value = tl.load($(vmul_2_ptr) + ((((r11_z / $(256)) % $(2048)) * $(256)) + (r11_z % $(256))), dtype=tl.float32)
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(524288)])
    tl.store($(vto_copy_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(163840))
    r12_value = tl.load($(vmul_3_ptr) + ((((r12_z / $(512)) % $(320)) * $(512)) + (r12_z % $(512))), dtype=tl.float32)
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(163840)])
    tl.store($(vto_copy_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(256))
    r13_value = tl.load($(vadd_1_ptr) + r13_z, dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(256)])
    tl.store($(vto_copy_4_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(1048576))
    r14_value = tl.load($(vmul_6_ptr) + ((((r14_z / $(2048)) % $(512)) * $(2048)) + (r14_z % $(2048))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(1048576)])
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
    r37_z = tl.arange($(917504))
    r37_value = tl.load($(vmul_5_ptr) + ((((r37_z / $(256)) % $(3584)) * $(256)) + (r37_z % $(256))), dtype=tl.float32)
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(917504)])
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
    r47_col = tl.arange($(2048))
    r47_lane = tl.arange($(256))
    r47_left = tl.load($(vto_copy_20_ptr) + ((((((r47_row[:, None] * $(256)) + r47_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r47_row[:, None] * $(256)) + r47_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r47_right = tl.load($(vto_copy_2_ptr) + (((r47_col[None, :] % $(2048)) * $(256)) + (r47_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r47_value = tl.dot(r47_left, r47_right)
    r47_value = r47_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r47_row[:, None] * $(2048)) + r47_col[None, :]), r47_value)
    r48_row = tl.arange($(128))
    r48_col = tl.arange($(320))
    r48_lane = tl.arange($(512))
    r48_left = tl.load($(input0_ptr) + ((((((r48_row[:, None] * $(512)) + r48_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r48_row[:, None] * $(512)) + r48_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r48_right = tl.load($(vto_copy_3_ptr) + (((r48_col[None, :] % $(320)) * $(512)) + (r48_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r48_value = tl.dot(r48_left, r48_right)
    r48_value = r48_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r48_row[:, None] * $(320)) + r48_col[None, :]), r48_value)
    r49_z = tl.arange($(32768))
    r49_value = tl.load($(vmm_2_ptr) + ((((r49_z / $(256)) % $(128)) * $(320)) + (r49_z % $(256))), dtype=tl.bfloat16)
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(32768)])
    tl.store($(vto_copy_21_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(32768))
    r50_value = ((tl.load($(vto_copy_21_ptr) + ((((r50_z / $(256)) % $(128)) * $(256)) + (r50_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_21_ptr) + ((((r50_z / $(256)) % $(128)) * $(256)) + (r50_z % $(256))), dtype=tl.float32)))
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(32768)])
    tl.store($(vpow_3_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(128))
    r51_lane = tl.arange($(256))
    r51_value = (tl.sum(tl.load($(vpow_3_ptr) + ((r51_z[:, None] * $(256)) + r51_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
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
    r54_z = tl.arange($(32768))
    r54_value = (tl.load($(vto_copy_21_ptr) + ((((r54_z / $(256)) % $(128)) * $(256)) + (r54_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r54_z / $(256)) % $(128)), dtype=tl.float32))
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(32768)])
    tl.store($(vmul_11_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(32768))
    r55_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r55_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_11_ptr) + ((((r55_z / $(256)) % $(128)) * $(256)) + (r55_z % $(256))), dtype=tl.float32))
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(32768)])
    tl.store($(vmul_12_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(32768))
    r56_value = tl.load($(vmul_12_ptr) + ((((r56_z / $(256)) % $(128)) * $(256)) + (r56_z % $(256))), dtype=tl.float32)
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(32768)])
    tl.store($(vto_copy_22_ptr) + r56_z, r56_value)
    r57_row = tl.arange($(128))
    r57_col = tl.arange($(3584))
    r57_lane = tl.arange($(256))
    r57_left = tl.load($(vto_copy_22_ptr) + ((((((r57_row[:, None] * $(256)) + r57_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r57_row[:, None] * $(256)) + r57_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r57_right = tl.load($(vto_copy_18_ptr) + (((r57_col[None, :] % $(3584)) * $(256)) + (r57_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r57_value = tl.dot(r57_left, r57_right)
    r57_value = r57_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r57_row[:, None] * $(3584)) + r57_col[None, :]), r57_value)
    r58_z = tl.arange($(32768))
    r58_value = (tl.load($(vmm_1_ptr) + ((((r58_z / $(32)) % $(128)) * $(2048)) + (((r58_z / $(4096)) % $(8)) * $(256)) + ((r58_z % $(32)) * $(2)) + $(192)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r58_z / $(32)) % $(128)) * $(64)) + (r58_z % $(32))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(32768)])
    tl.store($(vmul_13_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(32768))
    r59_value = (tl.load($(vmm_1_ptr) + ((((r59_z / $(32)) % $(128)) * $(2048)) + (((r59_z / $(4096)) % $(8)) * $(256)) + ((r59_z % $(32)) * $(2)) + $(193)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r59_z / $(32)) % $(128)) * $(64)) + (r59_z % $(32))), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(32768)])
    tl.store($(vmul_14_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(32768))
    r60_value = (tl.load($(vmul_13_ptr) + ((((r60_z / $(4096)) % $(8)) * $(4096)) + (((r60_z / $(32)) % $(128)) * $(32)) + (r60_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_14_ptr) + ((((r60_z / $(4096)) % $(8)) * $(4096)) + (((r60_z / $(32)) % $(128)) * $(32)) + (r60_z % $(32))), dtype=tl.bfloat16))
    r60_value = r60_value.to(tl.bfloat16)
    r60_value = tl.broadcast_to(r60_value, [$(32768)])
    tl.store($(vsub_4_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(32768))
    r61_value = (tl.load($(vmm_1_ptr) + ((((r61_z / $(32)) % $(128)) * $(2048)) + (((r61_z / $(4096)) % $(8)) * $(256)) + ((r61_z % $(32)) * $(2)) + $(193)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r61_z / $(32)) % $(128)) * $(64)) + (r61_z % $(32))), dtype=tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(32768)])
    tl.store($(vmul_15_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(32768))
    r62_value = (tl.load($(vmm_1_ptr) + ((((r62_z / $(32)) % $(128)) * $(2048)) + (((r62_z / $(4096)) % $(8)) * $(256)) + ((r62_z % $(32)) * $(2)) + $(192)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r62_z / $(32)) % $(128)) * $(64)) + (r62_z % $(32))), dtype=tl.bfloat16))
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
    r65_value = (tl.load($(vmm_2_ptr) + ((((r65_z / $(32)) % $(128)) * $(320)) + ((r65_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r65_z / $(32)) % $(128)) * $(64)) + (r65_z % $(32))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(4096)])
    tl.store($(vmul_17_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(4096))
    r66_value = (tl.load($(vmm_2_ptr) + ((((r66_z / $(32)) % $(128)) * $(320)) + ((r66_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r66_z / $(32)) % $(128)) * $(64)) + (r66_z % $(32))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(4096)])
    tl.store($(vmul_18_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(4096))
    r67_value = (tl.load($(vmul_17_ptr) + ((((r67_z / $(32)) % $(128)) * $(32)) + (r67_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_18_ptr) + ((((r67_z / $(32)) % $(128)) * $(32)) + (r67_z % $(32))), dtype=tl.bfloat16))
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(4096)])
    tl.store($(vsub_5_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(4096))
    r68_value = (tl.load($(vmm_2_ptr) + ((((r68_z / $(32)) % $(128)) * $(320)) + ((r68_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r68_z / $(32)) % $(128)) * $(64)) + (r68_z % $(32))), dtype=tl.bfloat16))
    r68_value = r68_value.to(tl.bfloat16)
    r68_value = tl.broadcast_to(r68_value, [$(4096)])
    tl.store($(vmul_19_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(4096))
    r69_value = (tl.load($(vmm_2_ptr) + ((((r69_z / $(32)) % $(128)) * $(320)) + ((r69_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r69_z / $(32)) % $(128)) * $(64)) + (r69_z % $(32))), dtype=tl.bfloat16))
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
    r72_z = tl.arange($(262144))
    r72_value = tl.where((r72_z % $(256)) < $(192), tl.load($(vmm_1_ptr) + ((((((((r72_z / $(256)) % $(128)) * $(2048)) + (((r72_z / $(32768)) % $(8)) * $(256)) + ((r72_z % $(256)) - $(0))) / $(2048)) % $(128)) * $(2048)) + (((((r72_z / $(256)) % $(128)) * $(2048)) + (((r72_z / $(32768)) % $(8)) * $(256)) + ((r72_z % $(256)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_4_ptr) + ((((r72_z / $(32768)) % $(8)) * $(8192)) + (((r72_z / $(256)) % $(128)) * $(64)) + (((r72_z % $(256)) - $(192)) % $(64))), dtype=tl.bfloat16))
    r72_value = r72_value.to(tl.bfloat16)
    r72_value = tl.broadcast_to(r72_value, [$(262144)])
    tl.store($(vcat_6_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(262144))
    r73_value = tl.where((r73_z % $(256)) < $(192), tl.load($(vmm_3_ptr) + ((((((((r73_z / $(256)) % $(128)) * $(3584)) + (((r73_z / $(32768)) % $(8)) * $(448)) + ((r73_z % $(256)) - $(0))) / $(3584)) % $(128)) * $(3584)) + (((((r73_z / $(256)) % $(128)) * $(3584)) + (((r73_z / $(32768)) % $(8)) * $(448)) + ((r73_z % $(256)) - $(0))) % $(3584))), dtype=tl.bfloat16), tl.load($(vcat_5_ptr) + ((((r73_z / $(256)) % $(128)) * $(64)) + (((r73_z % $(256)) - $(192)) % $(64))), dtype=tl.bfloat16))
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(262144)])
    tl.store($(vcat_7_ptr) + r73_z, r73_value)
    r74_row = tl.arange($(128))
    r74_col = tl.arange($(128))
    r74_lane = tl.arange($(256))
    for r74_group in range($(8)) {
      r74_left = tl.load($(vcat_6_ptr) + (((((((r74_group % $(8)) * $(32768)) + (r74_row[:, None] * $(256)) + r74_lane[None, :]) / $(32768)) % $(8)) * $(32768)) + ((((((r74_group % $(8)) * $(32768)) + (r74_row[:, None] * $(256)) + r74_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r74_group % $(8)) * $(32768)) + (r74_row[:, None] * $(256)) + r74_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r74_right = tl.load($(vcat_7_ptr) + (((((((r74_group % $(8)) * $(32768)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) / $(32768)) % $(8)) * $(32768)) + (((((r74_group % $(8)) * $(32768)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) % $(128)) * $(256)) + (((((r74_group % $(8)) * $(32768)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r74_value = tl.dot(r74_left, r74_right)
      r74_value = r74_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r74_group * $(16384)) + (r74_row[:, None] * $(128)) + r74_col[None, :]), r74_value)
    }
    r75_z = tl.arange($(131072))
    r75_value = (tl.load($(vbmm_ptr) + ((((r75_z / $(16384)) % $(8)) * $(16384)) + (((r75_z / $(128)) % $(128)) * $(128)) + (r75_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.0625, tl.bfloat16))
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
    r80_col = tl.arange($(2048))
    r80_lane = tl.arange($(512))
    r80_left = tl.load($(input1_ptr) + ((((((r80_row[:, None] * $(512)) + r80_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r80_row[:, None] * $(512)) + r80_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r80_right = tl.load($(vto_copy_6_ptr) + (((r80_lane[:, None] % $(512)) * $(2048)) + (r80_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r80_value = tl.dot(r80_left, r80_right)
    r80_value = r80_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r80_row[:, None] * $(2048)) + r80_col[None, :]), r80_value)
    r81_row = tl.arange($(128))
    r81_col = tl.arange($(256))
    r81_lane = tl.arange($(128))
    for r81_group in range($(8)) {
      r81_left = tl.load($(vto_copy_24_ptr) + (((((((r81_group % $(8)) * $(16384)) + (r81_lane[None, :] * $(128)) + r81_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r81_group % $(8)) * $(16384)) + (r81_lane[None, :] * $(128)) + r81_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r81_group % $(8)) * $(16384)) + (r81_lane[None, :] * $(128)) + r81_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r81_right = tl.load($(vmm_5_ptr) + (((((((r81_group % $(8)) * $(32768)) + (r81_lane[:, None] * $(256)) + r81_col[None, :]) / $(256)) % $(128)) * $(2048)) + ((((((r81_group % $(8)) * $(32768)) + (r81_lane[:, None] * $(256)) + r81_col[None, :]) / $(32768)) % $(8)) * $(256)) + ((((r81_group % $(8)) * $(32768)) + (r81_lane[:, None] * $(256)) + r81_col[None, :]) % $(256))), dtype=tl.bfloat16)
      r81_value = tl.dot(r81_left, r81_right)
      r81_value = r81_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r81_group * $(32768)) + (r81_row[:, None] * $(256)) + r81_col[None, :]), r81_value)
    }
    r82_row = tl.arange($(128))
    r82_col = tl.arange($(128))
    r82_lane = tl.arange($(256))
    for r82_group in range($(8)) {
      r82_left = tl.load($(vmm_5_ptr) + (((((((r82_group % $(8)) * $(32768)) + (r82_row[:, None] * $(256)) + r82_lane[None, :]) / $(256)) % $(128)) * $(2048)) + ((((((r82_group % $(8)) * $(32768)) + (r82_row[:, None] * $(256)) + r82_lane[None, :]) / $(32768)) % $(8)) * $(256)) + ((((r82_group % $(8)) * $(32768)) + (r82_row[:, None] * $(256)) + r82_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r82_right = tl.load($(vmm_3_ptr) + (((((((r82_group % $(8)) * $(32768)) + (r82_col[None, :] * $(256)) + r82_lane[:, None]) / $(256)) % $(128)) * $(3584)) + ((((((r82_group % $(8)) * $(32768)) + (r82_col[None, :] * $(256)) + r82_lane[:, None]) / $(32768)) % $(8)) * $(448)) + ((((r82_group % $(8)) * $(32768)) + (r82_col[None, :] * $(256)) + r82_lane[:, None]) % $(256)) + $(192)), dtype=tl.bfloat16)
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
    r86_value = (tl.load($(vto_copy_26_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(128)) % $(128)) * $(128)) + (r86_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.0625, tl.bfloat16))
    r86_value = r86_value.to(tl.bfloat16)
    r86_value = tl.broadcast_to(r86_value, [$(131072)])
    tl.store($(vmul_22_ptr) + r86_z, r86_value)
    r87_row = tl.arange($(256))
    r87_col = tl.arange($(128))
    r87_lane = tl.arange($(128))
    for r87_group in range($(8)) {
      r87_left = tl.load($(vcat_6_ptr) + (((((((r87_group % $(8)) * $(32768)) + (r87_lane[None, :] * $(256)) + r87_row[:, None]) / $(32768)) % $(8)) * $(32768)) + ((((((r87_group % $(8)) * $(32768)) + (r87_lane[None, :] * $(256)) + r87_row[:, None]) / $(256)) % $(128)) * $(256)) + ((((r87_group % $(8)) * $(32768)) + (r87_lane[None, :] * $(256)) + r87_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r87_right = tl.load($(vmul_22_ptr) + (((((((r87_group % $(8)) * $(16384)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r87_group % $(8)) * $(16384)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r87_group % $(8)) * $(16384)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r87_value = tl.dot(r87_left, r87_right)
      r87_value = r87_value.to(tl.bfloat16)
      tl.store($(vbmm_4_ptr) + ((r87_group * $(32768)) + (r87_row[:, None] * $(128)) + r87_col[None, :]), r87_value)
    }
    r88_z = tl.arange($(458752))
    r88_value = tl.where((r88_z % $(448)) < $(192), tl.load($(vbmm_4_ptr) + ((((((((r88_z / $(57344)) % $(8)) * $(32768)) + (((r88_z % $(448)) - $(0)) * $(128)) + ((r88_z / $(448)) % $(128))) / $(32768)) % $(8)) * $(32768)) + ((((((r88_z / $(57344)) % $(8)) * $(256)) + ((r88_z % $(448)) - $(0))) % $(256)) * $(128)) + ((r88_z / $(448)) % $(128))), dtype=tl.bfloat16), tl.load($(vbmm_2_ptr) + ((((((((r88_z / $(57344)) % $(8)) * $(32768)) + (((r88_z / $(448)) % $(128)) * $(256)) + ((r88_z % $(448)) - $(192))) / $(32768)) % $(8)) * $(32768)) + (((((((r88_z / $(57344)) % $(8)) * $(32768)) + (((r88_z / $(448)) % $(128)) * $(256)) + ((r88_z % $(448)) - $(192))) / $(256)) % $(128)) * $(256)) + (((((r88_z / $(57344)) % $(8)) * $(32768)) + (((r88_z / $(448)) % $(128)) * $(256)) + ((r88_z % $(448)) - $(192))) % $(256))), dtype=tl.bfloat16))
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(458752)])
    tl.store($(vcat_8_ptr) + r88_z, r88_value)
    r89_row = tl.arange($(3584))
    r89_col = tl.arange($(256))
    r89_lane = tl.arange($(128))
    r89_left = tl.load($(vcat_8_ptr) + ((((((((((r89_lane[None, :] * $(3584)) + r89_row[:, None]) / $(3584)) % $(128)) * $(3584)) + (((r89_lane[None, :] * $(3584)) + r89_row[:, None]) % $(3584))) / $(448)) % $(8)) * $(57344)) + (((((r89_lane[None, :] * $(3584)) + r89_row[:, None]) / $(3584)) % $(128)) * $(448)) + (((((((r89_lane[None, :] * $(3584)) + r89_row[:, None]) / $(3584)) % $(128)) * $(3584)) + (((r89_lane[None, :] * $(3584)) + r89_row[:, None]) % $(3584))) % $(448))), dtype=tl.bfloat16)
    r89_right = tl.load($(vto_copy_22_ptr) + ((((((r89_lane[:, None] * $(256)) + r89_col[None, :]) / $(256)) % $(128)) * $(256)) + (((r89_lane[:, None] * $(256)) + r89_col[None, :]) % $(256))), dtype=tl.bfloat16)
    r89_value = tl.dot(r89_left, r89_right)
    r89_value = r89_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r89_row[:, None] * $(256)) + r89_col[None, :]), r89_value)
    z = tl.arange($(917504))
    value = tl.load($(vmm_6_ptr) + ((((z / $(256)) % $(3584)) * $(256)) + (z % $(256))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(917504)])
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
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vbmm_4_ptr : RegionName := "vbmm_4"
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
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vsub_6_ptr : RegionName := "candidate_vsub_6"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_7_ptr : RegionName := "candidate_vsub_7"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vsub_8_ptr : RegionName := "candidate_vsub_8"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vsoftmax_backward_data_ptr : RegionName := "candidate_vsoftmax_backward_data"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vsoftmax_backward_data_1_ptr : RegionName := "candidate_vsoftmax_backward_data_1"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vsoftmax_backward_data_2_ptr : RegionName := "candidate_vsoftmax_backward_data_2"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
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
    r3_z = tl.arange($(524288))
    r3_value = (tl.load($(parameter2_ptr) + ((((r3_z / $(256)) % $(2048)) * $(256)) + (r3_z % $(256))), dtype=tl.float32) * 0.0625)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(524288)])
    tl.store($(candidate_vmul_2_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(163840))
    r4_value = (tl.load($(parameter3_ptr) + ((((r4_z / $(512)) % $(320)) * $(512)) + (r4_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(163840)])
    tl.store($(candidate_vmul_3_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(256))
    r5_value = (tl.load($(parameter4_ptr) + r5_z, dtype=tl.float32) * 0.1)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(256)])
    tl.store($(candidate_vmul_4_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(256))
    r6_value = (tl.load($(candidate_vmul_4_ptr) + r6_z, dtype=tl.float32) + 1.0)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(256)])
    tl.store($(candidate_vadd_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(917504))
    r7_value = (tl.load($(parameter5_ptr) + ((((r7_z / $(256)) % $(3584)) * $(256)) + (r7_z % $(256))), dtype=tl.float32) * 0.0625)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(917504)])
    tl.store($(candidate_vmul_5_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(1048576))
    r8_value = (tl.load($(parameter6_ptr) + ((((r8_z / $(2048)) % $(512)) * $(2048)) + (r8_z % $(2048))), dtype=tl.float32) * 0.02209708691207961)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(1048576)])
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
    r11_z = tl.arange($(524288))
    r11_value = tl.load($(candidate_vmul_2_ptr) + ((((r11_z / $(256)) % $(2048)) * $(256)) + (r11_z % $(256))), dtype=tl.float32)
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(524288)])
    tl.store($(candidate_vto_copy_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(163840))
    r12_value = tl.load($(candidate_vmul_3_ptr) + ((((r12_z / $(512)) % $(320)) * $(512)) + (r12_z % $(512))), dtype=tl.float32)
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(163840)])
    tl.store($(candidate_vto_copy_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(256))
    r13_value = tl.load($(candidate_vadd_1_ptr) + r13_z, dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(256)])
    tl.store($(candidate_vto_copy_4_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(1048576))
    r14_value = tl.load($(candidate_vmul_6_ptr) + ((((r14_z / $(2048)) % $(512)) * $(2048)) + (r14_z % $(2048))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(1048576)])
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
    r28_z = tl.arange($(917504))
    r28_value = tl.load($(candidate_vmul_5_ptr) + ((((r28_z / $(256)) % $(3584)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(917504)])
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
    r38_col = tl.arange($(2048))
    r38_lane = tl.arange($(256))
    r38_left = tl.load($(candidate_vto_copy_20_ptr) + ((((((r38_row[:, None] * $(256)) + r38_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r38_row[:, None] * $(256)) + r38_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r38_right = tl.load($(candidate_vto_copy_2_ptr) + (((r38_col[None, :] % $(2048)) * $(256)) + (r38_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r38_value = tl.dot(r38_left, r38_right)
    r38_value = r38_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r38_row[:, None] * $(2048)) + r38_col[None, :]), r38_value)
    r39_row = tl.arange($(128))
    r39_col = tl.arange($(320))
    r39_lane = tl.arange($(512))
    r39_left = tl.load($(input0_ptr) + ((((((r39_row[:, None] * $(512)) + r39_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r39_row[:, None] * $(512)) + r39_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r39_right = tl.load($(candidate_vto_copy_3_ptr) + (((r39_col[None, :] % $(320)) * $(512)) + (r39_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r39_value = tl.dot(r39_left, r39_right)
    r39_value = r39_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r39_row[:, None] * $(320)) + r39_col[None, :]), r39_value)
    r40_z = tl.arange($(32768))
    r40_value = tl.load($(candidate_vmm_2_ptr) + ((((r40_z / $(256)) % $(128)) * $(320)) + (r40_z % $(256))), dtype=tl.bfloat16)
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(32768)])
    tl.store($(candidate_vto_copy_21_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(32768))
    r41_value = ((tl.load($(candidate_vto_copy_21_ptr) + ((((r41_z / $(256)) % $(128)) * $(256)) + (r41_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_21_ptr) + ((((r41_z / $(256)) % $(128)) * $(256)) + (r41_z % $(256))), dtype=tl.float32)))
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(32768)])
    tl.store($(candidate_vpow_3_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(128))
    r42_lane = tl.arange($(256))
    r42_value = (tl.sum(tl.load($(candidate_vpow_3_ptr) + ((r42_z[:, None] * $(256)) + r42_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
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
    r45_z = tl.arange($(32768))
    r45_value = (tl.load($(candidate_vto_copy_21_ptr) + ((((r45_z / $(256)) % $(128)) * $(256)) + (r45_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r45_z / $(256)) % $(128)), dtype=tl.float32))
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(32768)])
    tl.store($(candidate_vmul_11_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(32768))
    r46_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r46_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_11_ptr) + ((((r46_z / $(256)) % $(128)) * $(256)) + (r46_z % $(256))), dtype=tl.float32))
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(32768)])
    tl.store($(candidate_vmul_12_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(32768))
    r47_value = tl.load($(candidate_vmul_12_ptr) + ((((r47_z / $(256)) % $(128)) * $(256)) + (r47_z % $(256))), dtype=tl.float32)
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(32768)])
    tl.store($(candidate_vto_copy_22_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(32768))
    r48_value = (tl.load($(candidate_vmm_1_ptr) + ((((r48_z / $(32)) % $(128)) * $(2048)) + (((r48_z / $(4096)) % $(8)) * $(256)) + ((r48_z % $(32)) * $(2)) + $(192)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r48_z / $(32)) % $(128)) * $(64)) + (r48_z % $(32))), dtype=tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(32768)])
    tl.store($(candidate_vmul_13_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(32768))
    r49_value = (tl.load($(candidate_vmm_1_ptr) + ((((r49_z / $(32)) % $(128)) * $(2048)) + (((r49_z / $(4096)) % $(8)) * $(256)) + ((r49_z % $(32)) * $(2)) + $(193)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r49_z / $(32)) % $(128)) * $(64)) + (r49_z % $(32))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(32768)])
    tl.store($(candidate_vmul_14_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(32768))
    r50_value = (tl.load($(candidate_vmul_13_ptr) + ((((r50_z / $(4096)) % $(8)) * $(4096)) + (((r50_z / $(32)) % $(128)) * $(32)) + (r50_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_14_ptr) + ((((r50_z / $(4096)) % $(8)) * $(4096)) + (((r50_z / $(32)) % $(128)) * $(32)) + (r50_z % $(32))), dtype=tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(32768)])
    tl.store($(candidate_vsub_4_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(32768))
    r51_value = (tl.load($(candidate_vmm_1_ptr) + ((((r51_z / $(32)) % $(128)) * $(2048)) + (((r51_z / $(4096)) % $(8)) * $(256)) + ((r51_z % $(32)) * $(2)) + $(193)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r51_z / $(32)) % $(128)) * $(64)) + (r51_z % $(32))), dtype=tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(32768)])
    tl.store($(candidate_vmul_15_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(32768))
    r52_value = (tl.load($(candidate_vmm_1_ptr) + ((((r52_z / $(32)) % $(128)) * $(2048)) + (((r52_z / $(4096)) % $(8)) * $(256)) + ((r52_z % $(32)) * $(2)) + $(192)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r52_z / $(32)) % $(128)) * $(64)) + (r52_z % $(32))), dtype=tl.bfloat16))
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
    r55_value = (tl.load($(candidate_vmm_2_ptr) + ((((r55_z / $(32)) % $(128)) * $(320)) + ((r55_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r55_z / $(32)) % $(128)) * $(64)) + (r55_z % $(32))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(4096)])
    tl.store($(candidate_vmul_17_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(4096))
    r56_value = (tl.load($(candidate_vmm_2_ptr) + ((((r56_z / $(32)) % $(128)) * $(320)) + ((r56_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r56_z / $(32)) % $(128)) * $(64)) + (r56_z % $(32))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(4096)])
    tl.store($(candidate_vmul_18_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(4096))
    r57_value = (tl.load($(candidate_vmul_17_ptr) + ((((r57_z / $(32)) % $(128)) * $(32)) + (r57_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_18_ptr) + ((((r57_z / $(32)) % $(128)) * $(32)) + (r57_z % $(32))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(4096)])
    tl.store($(candidate_vsub_5_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(4096))
    r58_value = (tl.load($(candidate_vmm_2_ptr) + ((((r58_z / $(32)) % $(128)) * $(320)) + ((r58_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r58_z / $(32)) % $(128)) * $(64)) + (r58_z % $(32))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(4096)])
    tl.store($(candidate_vmul_19_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(4096))
    r59_value = (tl.load($(candidate_vmm_2_ptr) + ((((r59_z / $(32)) % $(128)) * $(320)) + ((r59_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r59_z / $(32)) % $(128)) * $(64)) + (r59_z % $(32))), dtype=tl.bfloat16))
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
    r62_col = tl.arange($(1536))
    r62_lane = tl.arange($(256))
    for r62_group in range($(1)) {
      r62_left = tl.load($(candidate_vto_copy_22_ptr) + ((((((r62_row[:, None] * $(256)) + r62_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r62_row[:, None] * $(256)) + r62_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r62_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r62_lane[:, None] * $(1536)) + r62_col[None, :]) / $(192)) % $(8)) * $(114688)) + ((((r62_lane[:, None] * $(1536)) + r62_col[None, :]) % $(192)) * $(256)) + ((((r62_lane[:, None] * $(1536)) + r62_col[None, :]) / $(1536)) % $(256))), dtype=tl.bfloat16)
      r62_value = tl.dot(r62_left, r62_right)
      r62_value = r62_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r62_group * $(196608)) + (r62_row[:, None] * $(1536)) + r62_col[None, :]), r62_value)
    }
    r63_row = tl.arange($(128))
    r63_col = tl.arange($(2048))
    r63_lane = tl.arange($(256))
    for r63_group in range($(1)) {
      r63_left = tl.load($(candidate_vto_copy_22_ptr) + ((((((r63_row[:, None] * $(256)) + r63_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r63_row[:, None] * $(256)) + r63_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r63_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r63_lane[:, None] * $(2048)) + r63_col[None, :]) / $(256)) % $(8)) * $(114688)) + ((((r63_lane[:, None] * $(2048)) + r63_col[None, :]) % $(256)) * $(256)) + ((((r63_lane[:, None] * $(2048)) + r63_col[None, :]) / $(2048)) % $(256)) + $(49152)), dtype=tl.bfloat16)
      r63_value = tl.dot(r63_left, r63_right)
      r63_value = r63_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_1_ptr) + ((r63_group * $(262144)) + (r63_row[:, None] * $(2048)) + r63_col[None, :]), r63_value)
    }
    r64_z = tl.arange($(262144))
    r64_value = tl.where((r64_z % $(256)) < $(192), tl.load($(candidate_vmm_1_ptr) + ((((((((r64_z / $(2048)) % $(128)) * $(1536)) + (((r64_z / $(256)) % $(8)) * $(192)) + ((r64_z % $(256)) - $(0))) / $(1536)) % $(128)) * $(2048)) + (((((((r64_z / $(2048)) % $(128)) * $(1536)) + (((r64_z / $(256)) % $(8)) * $(192)) + ((r64_z % $(256)) - $(0))) / $(192)) % $(8)) * $(256)) + (((((r64_z / $(2048)) % $(128)) * $(1536)) + (((r64_z / $(256)) % $(8)) * $(192)) + ((r64_z % $(256)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_4_ptr) + ((((((((r64_z / $(2048)) % $(128)) * $(512)) + (((r64_z / $(256)) % $(8)) * $(64)) + ((r64_z % $(256)) - $(192))) / $(64)) % $(8)) * $(8192)) + (((((((r64_z / $(2048)) % $(128)) * $(512)) + (((r64_z / $(256)) % $(8)) * $(64)) + ((r64_z % $(256)) - $(192))) / $(512)) % $(128)) * $(64)) + (((((r64_z / $(2048)) % $(128)) * $(512)) + (((r64_z / $(256)) % $(8)) * $(64)) + ((r64_z % $(256)) - $(192))) % $(64))), dtype=tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(262144)])
    tl.store($(candidate_vcat_6_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(262144))
    r65_value = tl.where((r65_z % $(256)) < $(192), tl.load($(candidate_vbmm_ptr) + ((((((((r65_z / $(2048)) % $(128)) * $(1536)) + (((r65_z / $(256)) % $(8)) * $(192)) + ((r65_z % $(256)) - $(0))) / $(1536)) % $(128)) * $(1536)) + (((((((r65_z / $(2048)) % $(128)) * $(1536)) + (((r65_z / $(256)) % $(8)) * $(192)) + ((r65_z % $(256)) - $(0))) / $(192)) % $(8)) * $(192)) + (((((r65_z / $(2048)) % $(128)) * $(1536)) + (((r65_z / $(256)) % $(8)) * $(192)) + ((r65_z % $(256)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r65_z / $(2048)) % $(128)) * $(64)) + ((r65_z % $(256)) - $(192))) / $(64)) % $(128)) * $(64)) + (((((r65_z / $(2048)) % $(128)) * $(64)) + ((r65_z % $(256)) - $(192))) % $(64))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(262144)])
    tl.store($(candidate_vcat_7_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(98304))
    r66_value = tl.load($(candidate_vcat_6_ptr) + ((((r66_z / $(256)) % $(48)) * $(2048)) + (((r66_z / $(12288)) % $(8)) * $(256)) + (r66_z % $(256))), dtype=tl.bfloat16)
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(98304)])
    tl.store($(candidate_vto_copy_23_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(98304))
    r67_value = tl.load($(candidate_vcat_7_ptr) + ((((r67_z / $(256)) % $(48)) * $(2048)) + (((r67_z / $(12288)) % $(8)) * $(256)) + (r67_z % $(256))), dtype=tl.bfloat16)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(98304)])
    tl.store($(candidate_vto_copy_24_ptr) + r67_z, r67_value)
    r68_row = tl.arange($(48))
    r68_col = tl.arange($(48))
    r68_lane = tl.arange($(256))
    for r68_group in range($(8)) {
      r68_left = tl.load($(candidate_vto_copy_23_ptr) + (((((((r68_group % $(8)) * $(12288)) + (r68_row[:, None] * $(256)) + r68_lane[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r68_group % $(8)) * $(12288)) + (r68_row[:, None] * $(256)) + r68_lane[None, :]) / $(256)) % $(48)) * $(256)) + ((((r68_group % $(8)) * $(12288)) + (r68_row[:, None] * $(256)) + r68_lane[None, :]) % $(256))), dtype=tl.float32)
      r68_right = tl.load($(candidate_vto_copy_24_ptr) + (((((((r68_group % $(8)) * $(12288)) + (r68_lane[:, None] * $(48)) + r68_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r68_group % $(8)) * $(12288)) + (r68_lane[:, None] * $(48)) + r68_col[None, :]) % $(48)) * $(256)) + (((((r68_group % $(8)) * $(12288)) + (r68_lane[:, None] * $(48)) + r68_col[None, :]) / $(48)) % $(256))), dtype=tl.float32)
      r68_value = tl.dot(r68_left, r68_right)
      r68_value = r68_value.to(tl.float32)
      tl.store($(candidate_vbmm_2_ptr) + ((r68_group * $(2304)) + (r68_row[:, None] * $(48)) + r68_col[None, :]), r68_value)
    }
    r69_z = tl.arange($(18432))
    r69_value = (tl.load($(candidate_vbmm_2_ptr) + ((((r69_z / $(2304)) % $(8)) * $(2304)) + (((r69_z / $(48)) % $(48)) * $(48)) + (r69_z % $(48))), dtype=tl.float32) * 0.0625)
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(18432)])
    tl.store($(candidate_vmul_21_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(48))
    r70_value = (tl.cast(r70_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r70_value = tl.broadcast_to(r70_value, [$(48)])
    tl.store($(candidate_vadd_6_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(48))
    r71_value = (tl.load($(candidate_vadd_6_ptr) + r71_z, dtype=tl.int64) - tl.full([], $(48), dtype=tl.int64))
    r71_value = tl.broadcast_to(r71_value, [$(48)])
    tl.store($(candidate_vsub_6_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(2304))
    r72_value = (tl.cast((r72_z % $(48)), tl.int64) > tl.load($(candidate_vsub_6_ptr) + ((r72_z / $(48)) % $(48)), dtype=tl.int64))
    r72_value = tl.broadcast_to(r72_value, [$(2304)])
    tl.store($(candidate_vgt_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(18432))
    r73_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r73_z / $(48)) % $(48)) * $(48)) + (r73_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_21_ptr) + ((((r73_z / $(2304)) % $(8)) * $(2304)) + (((r73_z / $(48)) % $(48)) * $(48)) + (r73_z % $(48))), dtype=tl.float32))
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(18432))
    r74_lane = tl.arange($(48))
    r74_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r74_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r74_z[:, None] / $(48)) % $(48)) * $(48)) + (r74_lane[None, :] % $(48))), dtype=tl.float32)
    r74_maximum = tl.max(r74_row_values, axis=1, keep_dims=true)
    r74_weights = tl.exp(r74_row_values - r74_maximum)
    r74_total = tl.sum(r74_weights, axis=1)
    r74_maximum = tl.max(r74_row_values, axis=1)
    r74_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r74_z / $(2304)) % $(8)) * $(2304)) + (((r74_z / $(48)) % $(48)) * $(48)) + (r74_z % $(48))), dtype=tl.float32) - r74_maximum) / r74_total
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(18432)])
    tl.store($(candidate_vsoftmax_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(18432))
    r75_value = tl.load($(candidate_vsoftmax_ptr) + ((((r75_z / $(2304)) % $(8)) * $(2304)) + (((r75_z / $(48)) % $(48)) * $(48)) + (r75_z % $(48))), dtype=tl.float32)
    r75_value = r75_value.to(tl.bfloat16)
    r75_value = tl.broadcast_to(r75_value, [$(18432)])
    tl.store($(candidate_vto_copy_25_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(32768))
    r76_value = tl.load($(candidate_vcat_6_ptr) + ((((r76_z / $(256)) % $(16)) * $(2048)) + (((r76_z / $(4096)) % $(8)) * $(256)) + (r76_z % $(256)) + $(98304)), dtype=tl.bfloat16)
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(32768)])
    tl.store($(candidate_vto_copy_26_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(32768))
    r77_value = tl.load($(candidate_vcat_7_ptr) + ((((r77_z / $(256)) % $(16)) * $(2048)) + (((r77_z / $(4096)) % $(8)) * $(256)) + (r77_z % $(256)) + $(98304)), dtype=tl.bfloat16)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(32768)])
    tl.store($(candidate_vto_copy_27_ptr) + r77_z, r77_value)
    r78_row = tl.arange($(16))
    r78_col = tl.arange($(16))
    r78_lane = tl.arange($(256))
    for r78_group in range($(8)) {
      r78_left = tl.load($(candidate_vto_copy_26_ptr) + (((((((r78_group % $(8)) * $(4096)) + (r78_row[:, None] * $(256)) + r78_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r78_group % $(8)) * $(4096)) + (r78_row[:, None] * $(256)) + r78_lane[None, :]) / $(256)) % $(16)) * $(256)) + ((((r78_group % $(8)) * $(4096)) + (r78_row[:, None] * $(256)) + r78_lane[None, :]) % $(256))), dtype=tl.float32)
      r78_right = tl.load($(candidate_vto_copy_27_ptr) + (((((((r78_group % $(8)) * $(4096)) + (r78_lane[:, None] * $(16)) + r78_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r78_group % $(8)) * $(4096)) + (r78_lane[:, None] * $(16)) + r78_col[None, :]) % $(16)) * $(256)) + (((((r78_group % $(8)) * $(4096)) + (r78_lane[:, None] * $(16)) + r78_col[None, :]) / $(16)) % $(256))), dtype=tl.float32)
      r78_value = tl.dot(r78_left, r78_right)
      r78_value = r78_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r78_group * $(256)) + (r78_row[:, None] * $(16)) + r78_col[None, :]), r78_value)
    }
    r79_z = tl.arange($(2048))
    r79_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r79_z / $(256)) % $(8)) * $(256)) + (((r79_z / $(16)) % $(16)) * $(16)) + (r79_z % $(16))), dtype=tl.float32) * 0.0625)
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(2048)])
    tl.store($(candidate_vmul_22_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(16))
    r80_value = (tl.cast(r80_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r80_value = tl.broadcast_to(r80_value, [$(16)])
    tl.store($(candidate_vadd_7_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(16))
    r81_value = (tl.load($(candidate_vadd_7_ptr) + r81_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r81_value = tl.broadcast_to(r81_value, [$(16)])
    tl.store($(candidate_vsub_7_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(256))
    r82_value = (tl.cast((r82_z % $(16)), tl.int64) > tl.load($(candidate_vsub_7_ptr) + ((r82_z / $(16)) % $(16)), dtype=tl.int64))
    r82_value = tl.broadcast_to(r82_value, [$(256)])
    tl.store($(candidate_vgt_1_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(2048))
    r83_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r83_z / $(16)) % $(16)) * $(16)) + (r83_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_22_ptr) + ((((r83_z / $(256)) % $(8)) * $(256)) + (((r83_z / $(16)) % $(16)) * $(16)) + (r83_z % $(16))), dtype=tl.float32))
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(2048))
    r84_lane = tl.arange($(16))
    r84_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r84_z[:, None] / $(256)) % $(8)) * $(256)) + (((r84_z[:, None] / $(16)) % $(16)) * $(16)) + (r84_lane[None, :] % $(16))), dtype=tl.float32)
    r84_maximum = tl.max(r84_row_values, axis=1, keep_dims=true)
    r84_weights = tl.exp(r84_row_values - r84_maximum)
    r84_total = tl.sum(r84_weights, axis=1)
    r84_maximum = tl.max(r84_row_values, axis=1)
    r84_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r84_z / $(256)) % $(8)) * $(256)) + (((r84_z / $(16)) % $(16)) * $(16)) + (r84_z % $(16))), dtype=tl.float32) - r84_maximum) / r84_total
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(2048)])
    tl.store($(candidate_vsoftmax_1_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(2048))
    r85_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r85_z / $(256)) % $(8)) * $(256)) + (((r85_z / $(16)) % $(16)) * $(16)) + (r85_z % $(16))), dtype=tl.float32)
    r85_value = r85_value.to(tl.bfloat16)
    r85_value = tl.broadcast_to(r85_value, [$(2048)])
    tl.store($(candidate_vto_copy_28_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(131072))
    r86_value = tl.load($(candidate_vcat_6_ptr) + ((((r86_z / $(256)) % $(64)) * $(2048)) + (((r86_z / $(16384)) % $(8)) * $(256)) + (r86_z % $(256)) + $(131072)), dtype=tl.bfloat16)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(131072)])
    tl.store($(candidate_vto_copy_29_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(131072))
    r87_value = tl.load($(candidate_vcat_7_ptr) + ((((r87_z / $(256)) % $(64)) * $(2048)) + (((r87_z / $(16384)) % $(8)) * $(256)) + (r87_z % $(256)) + $(131072)), dtype=tl.bfloat16)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(131072)])
    tl.store($(candidate_vto_copy_30_ptr) + r87_z, r87_value)
    r88_row = tl.arange($(64))
    r88_col = tl.arange($(64))
    r88_lane = tl.arange($(256))
    for r88_group in range($(8)) {
      r88_left = tl.load($(candidate_vto_copy_29_ptr) + (((((((r88_group % $(8)) * $(16384)) + (r88_row[:, None] * $(256)) + r88_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r88_group % $(8)) * $(16384)) + (r88_row[:, None] * $(256)) + r88_lane[None, :]) / $(256)) % $(64)) * $(256)) + ((((r88_group % $(8)) * $(16384)) + (r88_row[:, None] * $(256)) + r88_lane[None, :]) % $(256))), dtype=tl.float32)
      r88_right = tl.load($(candidate_vto_copy_30_ptr) + (((((((r88_group % $(8)) * $(16384)) + (r88_lane[:, None] * $(64)) + r88_col[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r88_group % $(8)) * $(16384)) + (r88_lane[:, None] * $(64)) + r88_col[None, :]) % $(64)) * $(256)) + (((((r88_group % $(8)) * $(16384)) + (r88_lane[:, None] * $(64)) + r88_col[None, :]) / $(64)) % $(256))), dtype=tl.float32)
      r88_value = tl.dot(r88_left, r88_right)
      r88_value = r88_value.to(tl.float32)
      tl.store($(candidate_vbmm_6_ptr) + ((r88_group * $(4096)) + (r88_row[:, None] * $(64)) + r88_col[None, :]), r88_value)
    }
    r89_z = tl.arange($(32768))
    r89_value = (tl.load($(candidate_vbmm_6_ptr) + ((((r89_z / $(4096)) % $(8)) * $(4096)) + (((r89_z / $(64)) % $(64)) * $(64)) + (r89_z % $(64))), dtype=tl.float32) * 0.0625)
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(32768)])
    tl.store($(candidate_vmul_23_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(64))
    r90_value = (tl.cast(r90_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r90_value = tl.broadcast_to(r90_value, [$(64)])
    tl.store($(candidate_vadd_8_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(64))
    r91_value = (tl.load($(candidate_vadd_8_ptr) + r91_z, dtype=tl.int64) - tl.full([], $(64), dtype=tl.int64))
    r91_value = tl.broadcast_to(r91_value, [$(64)])
    tl.store($(candidate_vsub_8_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(4096))
    r92_value = (tl.cast((r92_z % $(64)), tl.int64) > tl.load($(candidate_vsub_8_ptr) + ((r92_z / $(64)) % $(64)), dtype=tl.int64))
    r92_value = tl.broadcast_to(r92_value, [$(4096)])
    tl.store($(candidate_vgt_2_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(32768))
    r93_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r93_z / $(64)) % $(64)) * $(64)) + (r93_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r93_z / $(4096)) % $(8)) * $(4096)) + (((r93_z / $(64)) % $(64)) * $(64)) + (r93_z % $(64))), dtype=tl.float32))
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(32768))
    r94_lane = tl.arange($(64))
    r94_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r94_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r94_z[:, None] / $(64)) % $(64)) * $(64)) + (r94_lane[None, :] % $(64))), dtype=tl.float32)
    r94_maximum = tl.max(r94_row_values, axis=1, keep_dims=true)
    r94_weights = tl.exp(r94_row_values - r94_maximum)
    r94_total = tl.sum(r94_weights, axis=1)
    r94_maximum = tl.max(r94_row_values, axis=1)
    r94_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r94_z / $(4096)) % $(8)) * $(4096)) + (((r94_z / $(64)) % $(64)) * $(64)) + (r94_z % $(64))), dtype=tl.float32) - r94_maximum) / r94_total
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(32768)])
    tl.store($(candidate_vsoftmax_2_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(32768))
    r95_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r95_z / $(4096)) % $(8)) * $(4096)) + (((r95_z / $(64)) % $(64)) * $(64)) + (r95_z % $(64))), dtype=tl.float32)
    r95_value = r95_value.to(tl.bfloat16)
    r95_value = tl.broadcast_to(r95_value, [$(32768)])
    tl.store($(candidate_vto_copy_31_ptr) + r95_z, r95_value)
    r96_row = tl.arange($(128))
    r96_col = tl.arange($(2048))
    r96_lane = tl.arange($(512))
    r96_left = tl.load($(input1_ptr) + ((((((r96_row[:, None] * $(512)) + r96_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r96_row[:, None] * $(512)) + r96_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r96_right = tl.load($(candidate_vto_copy_6_ptr) + (((r96_lane[:, None] % $(512)) * $(2048)) + (r96_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r96_value = tl.dot(r96_left, r96_right)
    r96_value = r96_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r96_row[:, None] * $(2048)) + r96_col[None, :]), r96_value)
    r97_row = tl.arange($(64))
    r97_col = tl.arange($(256))
    r97_lane = tl.arange($(64))
    for r97_group in range($(8)) {
      r97_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r97_group % $(8)) * $(4096)) + (r97_lane[None, :] * $(64)) + r97_row[:, None]) / $(4096)) % $(8)) * $(4096)) + ((((((r97_group % $(8)) * $(4096)) + (r97_lane[None, :] * $(64)) + r97_row[:, None]) / $(64)) % $(64)) * $(64)) + ((((r97_group % $(8)) * $(4096)) + (r97_lane[None, :] * $(64)) + r97_row[:, None]) % $(64))), dtype=tl.bfloat16)
      r97_right = tl.load($(candidate_vmm_4_ptr) + (((((((r97_group % $(8)) * $(16384)) + (r97_lane[:, None] * $(256)) + r97_col[None, :]) / $(256)) % $(64)) * $(2048)) + ((((((r97_group % $(8)) * $(16384)) + (r97_lane[:, None] * $(256)) + r97_col[None, :]) / $(16384)) % $(8)) * $(256)) + ((((r97_group % $(8)) * $(16384)) + (r97_lane[:, None] * $(256)) + r97_col[None, :]) % $(256)) + $(131072)), dtype=tl.bfloat16)
      r97_value = tl.dot(r97_left, r97_right)
      r97_value = r97_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_8_ptr) + ((r97_group * $(16384)) + (r97_row[:, None] * $(256)) + r97_col[None, :]), r97_value)
    }
    r98_row = tl.arange($(64))
    r98_col = tl.arange($(64))
    r98_lane = tl.arange($(256))
    for r98_group in range($(8)) {
      r98_left = tl.load($(candidate_vmm_4_ptr) + (((((((r98_group % $(8)) * $(16384)) + (r98_row[:, None] * $(256)) + r98_lane[None, :]) / $(256)) % $(64)) * $(2048)) + ((((((r98_group % $(8)) * $(16384)) + (r98_row[:, None] * $(256)) + r98_lane[None, :]) / $(16384)) % $(8)) * $(256)) + ((((r98_group % $(8)) * $(16384)) + (r98_row[:, None] * $(256)) + r98_lane[None, :]) % $(256)) + $(131072)), dtype=tl.bfloat16)
      r98_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r98_group % $(8)) * $(16384)) + (r98_col[None, :] * $(256)) + r98_lane[:, None]) / $(256)) % $(64)) * $(2048)) + ((((((r98_group % $(8)) * $(16384)) + (r98_col[None, :] * $(256)) + r98_lane[:, None]) / $(16384)) % $(8)) * $(256)) + ((((r98_group % $(8)) * $(16384)) + (r98_col[None, :] * $(256)) + r98_lane[:, None]) % $(256)) + $(131072)), dtype=tl.bfloat16)
      r98_value = tl.dot(r98_left, r98_right)
      r98_value = r98_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_9_ptr) + ((r98_group * $(4096)) + (r98_row[:, None] * $(64)) + r98_col[None, :]), r98_value)
    }
    r99_z = tl.arange($(32768))
    r99_value = tl.load($(candidate_vbmm_9_ptr) + ((((r99_z / $(4096)) % $(8)) * $(4096)) + (((r99_z / $(64)) % $(64)) * $(64)) + (r99_z % $(64))), dtype=tl.bfloat16)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(32768)])
    tl.store($(candidate_vto_copy_32_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(32768))
    r100_lane = tl.arange($(64))
    r100_row_values = tl.load($(candidate_vsoftmax_2_ptr) + ((((r100_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r100_z[:, None] / $(64)) % $(64)) * $(64)) + (r100_lane[None, :] % $(64))), dtype=tl.float32)
    r100_row_grad = tl.load($(candidate_vto_copy_32_ptr) + ((((r100_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r100_z[:, None] / $(64)) % $(64)) * $(64)) + (r100_lane[None, :] % $(64))), dtype=tl.float32)
    r100_weighted = r100_row_grad * r100_row_values
    r100_dot_grad = tl.sum(r100_weighted, axis=1)
    r100_value = (tl.load($(candidate_vto_copy_32_ptr) + ((((r100_z / $(4096)) % $(8)) * $(4096)) + (((r100_z / $(64)) % $(64)) * $(64)) + (r100_z % $(64))), dtype=tl.float32) - r100_dot_grad) * tl.load($(candidate_vsoftmax_2_ptr) + ((((r100_z / $(4096)) % $(8)) * $(4096)) + (((r100_z / $(64)) % $(64)) * $(64)) + (r100_z % $(64))), dtype=tl.float32)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(32768)])
    tl.store($(candidate_vsoftmax_backward_data_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(32768))
    r101_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r101_z / $(64)) % $(64)) * $(64)) + (r101_z % $(64))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_ptr) + ((((r101_z / $(4096)) % $(8)) * $(4096)) + (((r101_z / $(64)) % $(64)) * $(64)) + (r101_z % $(64))), dtype=tl.float32))
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(32768))
    r102_value = (tl.load($(candidate_vmasked_fill_3_ptr) + ((((r102_z / $(4096)) % $(8)) * $(4096)) + (((r102_z / $(64)) % $(64)) * $(64)) + (r102_z % $(64))), dtype=tl.float32) * 0.0625)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(32768)])
    tl.store($(candidate_vmul_24_ptr) + r102_z, r102_value)
    r103_row = tl.arange($(256))
    r103_col = tl.arange($(64))
    r103_lane = tl.arange($(64))
    for r103_group in range($(8)) {
      r103_left = tl.load($(candidate_vto_copy_29_ptr) + (((((((r103_group % $(8)) * $(16384)) + (r103_lane[None, :] * $(256)) + r103_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r103_group % $(8)) * $(16384)) + (r103_lane[None, :] * $(256)) + r103_row[:, None]) / $(256)) % $(64)) * $(256)) + ((((r103_group % $(8)) * $(16384)) + (r103_lane[None, :] * $(256)) + r103_row[:, None]) % $(256))), dtype=tl.float32)
      r103_right = tl.load($(candidate_vmul_24_ptr) + (((((((r103_group % $(8)) * $(4096)) + (r103_lane[:, None] * $(64)) + r103_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r103_group % $(8)) * $(4096)) + (r103_lane[:, None] * $(64)) + r103_col[None, :]) / $(64)) % $(64)) * $(64)) + ((((r103_group % $(8)) * $(4096)) + (r103_lane[:, None] * $(64)) + r103_col[None, :]) % $(64))), dtype=tl.float32)
      r103_value = tl.dot(r103_left, r103_right)
      r103_value = r103_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r103_group * $(16384)) + (r103_row[:, None] * $(64)) + r103_col[None, :]), r103_value)
    }
    r104_z = tl.arange($(262144))
    r104_value = tl.where(((((r104_z / $(2048)) % $(128)) >= $(64)) & (((r104_z / $(2048)) % $(128)) < $(128)) & ((((r104_z / $(2048)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_8_ptr) + ((((((((r104_z / $(256)) % $(8)) * $(16384)) + ((((r104_z / $(2048)) % $(128)) - $(64)) * $(256)) + (r104_z % $(256))) / $(16384)) % $(8)) * $(16384)) + ((((((r104_z / $(256)) % $(8)) * $(64)) + (((r104_z / $(2048)) % $(128)) - $(64))) % $(64)) * $(256)) + (r104_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(262144)])
    tl.store($(candidate_vslice_backward_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(131072))
    r105_value = tl.load($(candidate_vbmm_10_ptr) + ((((r105_z / $(16384)) % $(8)) * $(16384)) + ((r105_z % $(256)) * $(64)) + ((r105_z / $(256)) % $(64))), dtype=tl.float32)
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(131072)])
    tl.store($(candidate_vto_copy_33_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(262144))
    r106_value = tl.where(((((r106_z / $(2048)) % $(128)) >= $(64)) & (((r106_z / $(2048)) % $(128)) < $(128)) & ((((r106_z / $(2048)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_33_ptr) + ((((r106_z / $(256)) % $(8)) * $(16384)) + (((((r106_z / $(2048)) % $(128)) - $(64)) % $(64)) * $(256)) + (r106_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(262144)])
    tl.store($(candidate_vslice_backward_1_ptr) + r106_z, r106_value)
    r107_row = tl.arange($(16))
    r107_col = tl.arange($(256))
    r107_lane = tl.arange($(16))
    for r107_group in range($(8)) {
      r107_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r107_group % $(8)) * $(256)) + (r107_lane[None, :] * $(16)) + r107_row[:, None]) / $(256)) % $(8)) * $(256)) + ((((((r107_group % $(8)) * $(256)) + (r107_lane[None, :] * $(16)) + r107_row[:, None]) / $(16)) % $(16)) * $(16)) + ((((r107_group % $(8)) * $(256)) + (r107_lane[None, :] * $(16)) + r107_row[:, None]) % $(16))), dtype=tl.bfloat16)
      r107_right = tl.load($(candidate_vmm_4_ptr) + (((((((r107_group % $(8)) * $(4096)) + (r107_lane[:, None] * $(256)) + r107_col[None, :]) / $(256)) % $(16)) * $(2048)) + ((((((r107_group % $(8)) * $(4096)) + (r107_lane[:, None] * $(256)) + r107_col[None, :]) / $(4096)) % $(8)) * $(256)) + ((((r107_group % $(8)) * $(4096)) + (r107_lane[:, None] * $(256)) + r107_col[None, :]) % $(256)) + $(98304)), dtype=tl.bfloat16)
      r107_value = tl.dot(r107_left, r107_right)
      r107_value = r107_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_11_ptr) + ((r107_group * $(4096)) + (r107_row[:, None] * $(256)) + r107_col[None, :]), r107_value)
    }
    r108_row = tl.arange($(16))
    r108_col = tl.arange($(16))
    r108_lane = tl.arange($(256))
    for r108_group in range($(8)) {
      r108_left = tl.load($(candidate_vmm_4_ptr) + (((((((r108_group % $(8)) * $(4096)) + (r108_row[:, None] * $(256)) + r108_lane[None, :]) / $(256)) % $(16)) * $(2048)) + ((((((r108_group % $(8)) * $(4096)) + (r108_row[:, None] * $(256)) + r108_lane[None, :]) / $(4096)) % $(8)) * $(256)) + ((((r108_group % $(8)) * $(4096)) + (r108_row[:, None] * $(256)) + r108_lane[None, :]) % $(256)) + $(98304)), dtype=tl.bfloat16)
      r108_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r108_group % $(8)) * $(4096)) + (r108_col[None, :] * $(256)) + r108_lane[:, None]) / $(256)) % $(16)) * $(2048)) + ((((((r108_group % $(8)) * $(4096)) + (r108_col[None, :] * $(256)) + r108_lane[:, None]) / $(4096)) % $(8)) * $(256)) + ((((r108_group % $(8)) * $(4096)) + (r108_col[None, :] * $(256)) + r108_lane[:, None]) % $(256)) + $(98304)), dtype=tl.bfloat16)
      r108_value = tl.dot(r108_left, r108_right)
      r108_value = r108_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r108_group * $(256)) + (r108_row[:, None] * $(16)) + r108_col[None, :]), r108_value)
    }
    r109_z = tl.arange($(2048))
    r109_value = tl.load($(candidate_vbmm_12_ptr) + ((((r109_z / $(256)) % $(8)) * $(256)) + (((r109_z / $(16)) % $(16)) * $(16)) + (r109_z % $(16))), dtype=tl.bfloat16)
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(2048)])
    tl.store($(candidate_vto_copy_34_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(2048))
    r110_lane = tl.arange($(16))
    r110_row_values = tl.load($(candidate_vsoftmax_1_ptr) + ((((r110_z[:, None] / $(256)) % $(8)) * $(256)) + (((r110_z[:, None] / $(16)) % $(16)) * $(16)) + (r110_lane[None, :] % $(16))), dtype=tl.float32)
    r110_row_grad = tl.load($(candidate_vto_copy_34_ptr) + ((((r110_z[:, None] / $(256)) % $(8)) * $(256)) + (((r110_z[:, None] / $(16)) % $(16)) * $(16)) + (r110_lane[None, :] % $(16))), dtype=tl.float32)
    r110_weighted = r110_row_grad * r110_row_values
    r110_dot_grad = tl.sum(r110_weighted, axis=1)
    r110_value = (tl.load($(candidate_vto_copy_34_ptr) + ((((r110_z / $(256)) % $(8)) * $(256)) + (((r110_z / $(16)) % $(16)) * $(16)) + (r110_z % $(16))), dtype=tl.float32) - r110_dot_grad) * tl.load($(candidate_vsoftmax_1_ptr) + ((((r110_z / $(256)) % $(8)) * $(256)) + (((r110_z / $(16)) % $(16)) * $(16)) + (r110_z % $(16))), dtype=tl.float32)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(2048)])
    tl.store($(candidate_vsoftmax_backward_data_1_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(2048))
    r111_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r111_z / $(16)) % $(16)) * $(16)) + (r111_z % $(16))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_1_ptr) + ((((r111_z / $(256)) % $(8)) * $(256)) + (((r111_z / $(16)) % $(16)) * $(16)) + (r111_z % $(16))), dtype=tl.float32))
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(2048))
    r112_value = (tl.load($(candidate_vmasked_fill_4_ptr) + ((((r112_z / $(256)) % $(8)) * $(256)) + (((r112_z / $(16)) % $(16)) * $(16)) + (r112_z % $(16))), dtype=tl.float32) * 0.0625)
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(2048)])
    tl.store($(candidate_vmul_25_ptr) + r112_z, r112_value)
    r113_row = tl.arange($(256))
    r113_col = tl.arange($(16))
    r113_lane = tl.arange($(16))
    for r113_group in range($(8)) {
      r113_left = tl.load($(candidate_vto_copy_26_ptr) + (((((((r113_group % $(8)) * $(4096)) + (r113_lane[None, :] * $(256)) + r113_row[:, None]) / $(4096)) % $(8)) * $(4096)) + ((((((r113_group % $(8)) * $(4096)) + (r113_lane[None, :] * $(256)) + r113_row[:, None]) / $(256)) % $(16)) * $(256)) + ((((r113_group % $(8)) * $(4096)) + (r113_lane[None, :] * $(256)) + r113_row[:, None]) % $(256))), dtype=tl.float32)
      r113_right = tl.load($(candidate_vmul_25_ptr) + (((((((r113_group % $(8)) * $(256)) + (r113_lane[:, None] * $(16)) + r113_col[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r113_group % $(8)) * $(256)) + (r113_lane[:, None] * $(16)) + r113_col[None, :]) / $(16)) % $(16)) * $(16)) + ((((r113_group % $(8)) * $(256)) + (r113_lane[:, None] * $(16)) + r113_col[None, :]) % $(16))), dtype=tl.float32)
      r113_value = tl.dot(r113_left, r113_right)
      r113_value = r113_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r113_group * $(4096)) + (r113_row[:, None] * $(16)) + r113_col[None, :]), r113_value)
    }
    r114_z = tl.arange($(262144))
    r114_value = tl.where(((((r114_z / $(2048)) % $(128)) >= $(48)) & (((r114_z / $(2048)) % $(128)) < $(64)) & ((((r114_z / $(2048)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_11_ptr) + ((((((((r114_z / $(256)) % $(8)) * $(4096)) + ((((r114_z / $(2048)) % $(128)) - $(48)) * $(256)) + (r114_z % $(256))) / $(4096)) % $(8)) * $(4096)) + ((((((r114_z / $(256)) % $(8)) * $(16)) + (((r114_z / $(2048)) % $(128)) - $(48))) % $(16)) * $(256)) + (r114_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r114_value = r114_value.to(tl.bfloat16)
    r114_value = tl.broadcast_to(r114_value, [$(262144)])
    tl.store($(candidate_vslice_backward_2_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(262144))
    r115_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r115_z / $(2048)) % $(128)) * $(2048)) + (((r115_z / $(256)) % $(8)) * $(256)) + (r115_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_2_ptr) + ((((r115_z / $(2048)) % $(128)) * $(2048)) + (((r115_z / $(256)) % $(8)) * $(256)) + (r115_z % $(256))), dtype=tl.bfloat16))
    r115_value = r115_value.to(tl.bfloat16)
    r115_value = tl.broadcast_to(r115_value, [$(262144)])
    tl.store($(candidate_vadd_9_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(32768))
    r116_value = tl.load($(candidate_vbmm_13_ptr) + ((((r116_z / $(4096)) % $(8)) * $(4096)) + ((r116_z % $(256)) * $(16)) + ((r116_z / $(256)) % $(16))), dtype=tl.float32)
    r116_value = r116_value.to(tl.bfloat16)
    r116_value = tl.broadcast_to(r116_value, [$(32768)])
    tl.store($(candidate_vto_copy_35_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(262144))
    r117_value = tl.where(((((r117_z / $(2048)) % $(128)) >= $(48)) & (((r117_z / $(2048)) % $(128)) < $(64)) & ((((r117_z / $(2048)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_35_ptr) + ((((r117_z / $(256)) % $(8)) * $(4096)) + (((((r117_z / $(2048)) % $(128)) - $(48)) % $(16)) * $(256)) + (r117_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r117_value = r117_value.to(tl.bfloat16)
    r117_value = tl.broadcast_to(r117_value, [$(262144)])
    tl.store($(candidate_vslice_backward_3_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(262144))
    r118_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r118_z / $(2048)) % $(128)) * $(2048)) + (((r118_z / $(256)) % $(8)) * $(256)) + (r118_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_3_ptr) + ((((r118_z / $(2048)) % $(128)) * $(2048)) + (((r118_z / $(256)) % $(8)) * $(256)) + (r118_z % $(256))), dtype=tl.bfloat16))
    r118_value = r118_value.to(tl.bfloat16)
    r118_value = tl.broadcast_to(r118_value, [$(262144)])
    tl.store($(candidate_vadd_10_ptr) + r118_z, r118_value)
    r119_row = tl.arange($(48))
    r119_col = tl.arange($(256))
    r119_lane = tl.arange($(48))
    for r119_group in range($(8)) {
      r119_left = tl.load($(candidate_vto_copy_25_ptr) + (((((((r119_group % $(8)) * $(2304)) + (r119_lane[None, :] * $(48)) + r119_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r119_group % $(8)) * $(2304)) + (r119_lane[None, :] * $(48)) + r119_row[:, None]) / $(48)) % $(48)) * $(48)) + ((((r119_group % $(8)) * $(2304)) + (r119_lane[None, :] * $(48)) + r119_row[:, None]) % $(48))), dtype=tl.bfloat16)
      r119_right = tl.load($(candidate_vmm_4_ptr) + (((((((r119_group % $(8)) * $(12288)) + (r119_lane[:, None] * $(256)) + r119_col[None, :]) / $(256)) % $(48)) * $(2048)) + ((((((r119_group % $(8)) * $(12288)) + (r119_lane[:, None] * $(256)) + r119_col[None, :]) / $(12288)) % $(8)) * $(256)) + ((((r119_group % $(8)) * $(12288)) + (r119_lane[:, None] * $(256)) + r119_col[None, :]) % $(256))), dtype=tl.bfloat16)
      r119_value = tl.dot(r119_left, r119_right)
      r119_value = r119_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_14_ptr) + ((r119_group * $(12288)) + (r119_row[:, None] * $(256)) + r119_col[None, :]), r119_value)
    }
    r120_row = tl.arange($(48))
    r120_col = tl.arange($(48))
    r120_lane = tl.arange($(256))
    for r120_group in range($(8)) {
      r120_left = tl.load($(candidate_vmm_4_ptr) + (((((((r120_group % $(8)) * $(12288)) + (r120_row[:, None] * $(256)) + r120_lane[None, :]) / $(256)) % $(48)) * $(2048)) + ((((((r120_group % $(8)) * $(12288)) + (r120_row[:, None] * $(256)) + r120_lane[None, :]) / $(12288)) % $(8)) * $(256)) + ((((r120_group % $(8)) * $(12288)) + (r120_row[:, None] * $(256)) + r120_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r120_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r120_group % $(8)) * $(12288)) + (r120_col[None, :] * $(256)) + r120_lane[:, None]) / $(256)) % $(48)) * $(2048)) + ((((((r120_group % $(8)) * $(12288)) + (r120_col[None, :] * $(256)) + r120_lane[:, None]) / $(12288)) % $(8)) * $(256)) + ((((r120_group % $(8)) * $(12288)) + (r120_col[None, :] * $(256)) + r120_lane[:, None]) % $(256))), dtype=tl.bfloat16)
      r120_value = tl.dot(r120_left, r120_right)
      r120_value = r120_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_15_ptr) + ((r120_group * $(2304)) + (r120_row[:, None] * $(48)) + r120_col[None, :]), r120_value)
    }
    r121_z = tl.arange($(18432))
    r121_value = tl.load($(candidate_vbmm_15_ptr) + ((((r121_z / $(2304)) % $(8)) * $(2304)) + (((r121_z / $(48)) % $(48)) * $(48)) + (r121_z % $(48))), dtype=tl.bfloat16)
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(18432)])
    tl.store($(candidate_vto_copy_36_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(18432))
    r122_lane = tl.arange($(48))
    r122_row_values = tl.load($(candidate_vsoftmax_ptr) + ((((r122_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r122_z[:, None] / $(48)) % $(48)) * $(48)) + (r122_lane[None, :] % $(48))), dtype=tl.float32)
    r122_row_grad = tl.load($(candidate_vto_copy_36_ptr) + ((((r122_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r122_z[:, None] / $(48)) % $(48)) * $(48)) + (r122_lane[None, :] % $(48))), dtype=tl.float32)
    r122_weighted = r122_row_grad * r122_row_values
    r122_dot_grad = tl.sum(r122_weighted, axis=1)
    r122_value = (tl.load($(candidate_vto_copy_36_ptr) + ((((r122_z / $(2304)) % $(8)) * $(2304)) + (((r122_z / $(48)) % $(48)) * $(48)) + (r122_z % $(48))), dtype=tl.float32) - r122_dot_grad) * tl.load($(candidate_vsoftmax_ptr) + ((((r122_z / $(2304)) % $(8)) * $(2304)) + (((r122_z / $(48)) % $(48)) * $(48)) + (r122_z % $(48))), dtype=tl.float32)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(18432)])
    tl.store($(candidate_vsoftmax_backward_data_2_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(18432))
    r123_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r123_z / $(48)) % $(48)) * $(48)) + (r123_z % $(48))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_2_ptr) + ((((r123_z / $(2304)) % $(8)) * $(2304)) + (((r123_z / $(48)) % $(48)) * $(48)) + (r123_z % $(48))), dtype=tl.float32))
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(18432))
    r124_value = (tl.load($(candidate_vmasked_fill_5_ptr) + ((((r124_z / $(2304)) % $(8)) * $(2304)) + (((r124_z / $(48)) % $(48)) * $(48)) + (r124_z % $(48))), dtype=tl.float32) * 0.0625)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(18432)])
    tl.store($(candidate_vmul_26_ptr) + r124_z, r124_value)
    r125_row = tl.arange($(256))
    r125_col = tl.arange($(48))
    r125_lane = tl.arange($(48))
    for r125_group in range($(8)) {
      r125_left = tl.load($(candidate_vto_copy_23_ptr) + (((((((r125_group % $(8)) * $(12288)) + (r125_lane[None, :] * $(256)) + r125_row[:, None]) / $(12288)) % $(8)) * $(12288)) + ((((((r125_group % $(8)) * $(12288)) + (r125_lane[None, :] * $(256)) + r125_row[:, None]) / $(256)) % $(48)) * $(256)) + ((((r125_group % $(8)) * $(12288)) + (r125_lane[None, :] * $(256)) + r125_row[:, None]) % $(256))), dtype=tl.float32)
      r125_right = tl.load($(candidate_vmul_26_ptr) + (((((((r125_group % $(8)) * $(2304)) + (r125_lane[:, None] * $(48)) + r125_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r125_group % $(8)) * $(2304)) + (r125_lane[:, None] * $(48)) + r125_col[None, :]) / $(48)) % $(48)) * $(48)) + ((((r125_group % $(8)) * $(2304)) + (r125_lane[:, None] * $(48)) + r125_col[None, :]) % $(48))), dtype=tl.float32)
      r125_value = tl.dot(r125_left, r125_right)
      r125_value = r125_value.to(tl.float32)
      tl.store($(candidate_vbmm_16_ptr) + ((r125_group * $(12288)) + (r125_row[:, None] * $(48)) + r125_col[None, :]), r125_value)
    }
    r126_z = tl.arange($(262144))
    r126_value = tl.where(((((r126_z / $(2048)) % $(128)) >= $(0)) & (((r126_z / $(2048)) % $(128)) < $(48)) & ((((r126_z / $(2048)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_14_ptr) + ((((((((r126_z / $(256)) % $(8)) * $(12288)) + ((((r126_z / $(2048)) % $(128)) - $(0)) * $(256)) + (r126_z % $(256))) / $(12288)) % $(8)) * $(12288)) + ((((((r126_z / $(256)) % $(8)) * $(48)) + (((r126_z / $(2048)) % $(128)) - $(0))) % $(48)) * $(256)) + (r126_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r126_value = r126_value.to(tl.bfloat16)
    r126_value = tl.broadcast_to(r126_value, [$(262144)])
    tl.store($(candidate_vslice_backward_4_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(262144))
    r127_value = (tl.load($(candidate_vadd_9_ptr) + ((((r127_z / $(2048)) % $(128)) * $(2048)) + (((r127_z / $(256)) % $(8)) * $(256)) + (r127_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_4_ptr) + ((((r127_z / $(2048)) % $(128)) * $(2048)) + (((r127_z / $(256)) % $(8)) * $(256)) + (r127_z % $(256))), dtype=tl.bfloat16))
    r127_value = r127_value.to(tl.bfloat16)
    r127_value = tl.broadcast_to(r127_value, [$(262144)])
    tl.store($(candidate_vadd_11_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(98304))
    r128_value = tl.load($(candidate_vbmm_16_ptr) + ((((r128_z / $(12288)) % $(8)) * $(12288)) + ((r128_z % $(256)) * $(48)) + ((r128_z / $(256)) % $(48))), dtype=tl.float32)
    r128_value = r128_value.to(tl.bfloat16)
    r128_value = tl.broadcast_to(r128_value, [$(98304)])
    tl.store($(candidate_vto_copy_37_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(262144))
    r129_value = tl.where(((((r129_z / $(2048)) % $(128)) >= $(0)) & (((r129_z / $(2048)) % $(128)) < $(48)) & ((((r129_z / $(2048)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_37_ptr) + ((((r129_z / $(256)) % $(8)) * $(12288)) + (((((r129_z / $(2048)) % $(128)) - $(0)) % $(48)) * $(256)) + (r129_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r129_value = r129_value.to(tl.bfloat16)
    r129_value = tl.broadcast_to(r129_value, [$(262144)])
    tl.store($(candidate_vslice_backward_5_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(262144))
    r130_value = (tl.load($(candidate_vadd_10_ptr) + ((((r130_z / $(2048)) % $(128)) * $(2048)) + (((r130_z / $(256)) % $(8)) * $(256)) + (r130_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r130_z / $(2048)) % $(128)) * $(2048)) + (((r130_z / $(256)) % $(8)) * $(256)) + (r130_z % $(256))), dtype=tl.bfloat16))
    r130_value = r130_value.to(tl.bfloat16)
    r130_value = tl.broadcast_to(r130_value, [$(262144)])
    tl.store($(candidate_vadd_12_ptr) + r130_z, r130_value)
    r131_row = tl.arange($(256))
    r131_col = tl.arange($(2048))
    r131_lane = tl.arange($(128))
    for r131_group in range($(1)) {
      r131_left = tl.load($(candidate_vto_copy_22_ptr) + ((((((r131_lane[None, :] * $(256)) + r131_row[:, None]) / $(256)) % $(128)) * $(256)) + (((r131_lane[None, :] * $(256)) + r131_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r131_right = tl.load($(candidate_vadd_11_ptr) + ((((((r131_lane[:, None] * $(2048)) + r131_col[None, :]) / $(2048)) % $(128)) * $(2048)) + (((((r131_lane[:, None] * $(2048)) + r131_col[None, :]) / $(256)) % $(8)) * $(256)) + (((r131_lane[:, None] * $(2048)) + r131_col[None, :]) % $(256))), dtype=tl.bfloat16)
      r131_value = tl.dot(r131_left, r131_right)
      r131_value = r131_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_17_ptr) + ((r131_group * $(524288)) + (r131_row[:, None] * $(2048)) + r131_col[None, :]), r131_value)
    }
    r132_row = tl.arange($(256))
    r132_col = tl.arange($(1536))
    r132_lane = tl.arange($(128))
    for r132_group in range($(1)) {
      r132_left = tl.load($(candidate_vto_copy_22_ptr) + ((((((r132_lane[None, :] * $(256)) + r132_row[:, None]) / $(256)) % $(128)) * $(256)) + (((r132_lane[None, :] * $(256)) + r132_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r132_right = tl.load($(candidate_vadd_12_ptr) + ((((((r132_lane[:, None] * $(1536)) + r132_col[None, :]) / $(1536)) % $(128)) * $(2048)) + (((((r132_lane[:, None] * $(1536)) + r132_col[None, :]) / $(192)) % $(8)) * $(256)) + (((r132_lane[:, None] * $(1536)) + r132_col[None, :]) % $(192))), dtype=tl.bfloat16)
      r132_value = tl.dot(r132_left, r132_right)
      r132_value = r132_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_18_ptr) + ((r132_group * $(393216)) + (r132_row[:, None] * $(1536)) + r132_col[None, :]), r132_value)
    }
    r133_z = tl.arange($(917504))
    r133_value = tl.where(((((r133_z / $(256)) % $(448)) >= $(192)) & (((r133_z / $(256)) % $(448)) < $(448)) & ((((r133_z / $(256)) % $(448)) - $(192)) % $(1) == $(0))), tl.load($(candidate_vbmm_17_ptr) + (((r133_z % $(256)) * $(2048)) + (((((((r133_z / $(114688)) % $(8)) * $(65536)) + ((((r133_z / $(256)) % $(448)) - $(192)) * $(256)) + (r133_z % $(256))) / $(65536)) % $(8)) * $(256)) + (((((r133_z / $(114688)) % $(8)) * $(256)) + (((r133_z / $(256)) % $(448)) - $(192))) % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r133_value = r133_value.to(tl.bfloat16)
    r133_value = tl.broadcast_to(r133_value, [$(917504)])
    tl.store($(candidate_vslice_backward_6_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(917504))
    r134_value = tl.where(((((r134_z / $(256)) % $(448)) >= $(0)) & (((r134_z / $(256)) % $(448)) < $(192)) & ((((r134_z / $(256)) % $(448)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_18_ptr) + (((r134_z % $(256)) * $(1536)) + (((((((r134_z / $(114688)) % $(8)) * $(49152)) + ((((r134_z / $(256)) % $(448)) - $(0)) * $(256)) + (r134_z % $(256))) / $(49152)) % $(8)) * $(192)) + (((((r134_z / $(114688)) % $(8)) * $(192)) + (((r134_z / $(256)) % $(448)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r134_value = r134_value.to(tl.bfloat16)
    r134_value = tl.broadcast_to(r134_value, [$(917504)])
    tl.store($(candidate_vslice_backward_7_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(917504))
    r135_value = (tl.load($(candidate_vslice_backward_6_ptr) + ((((r135_z / $(114688)) % $(8)) * $(114688)) + (((r135_z / $(256)) % $(448)) * $(256)) + (r135_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_7_ptr) + ((((r135_z / $(114688)) % $(8)) * $(114688)) + (((r135_z / $(256)) % $(448)) * $(256)) + (r135_z % $(256))), dtype=tl.bfloat16))
    r135_value = r135_value.to(tl.bfloat16)
    r135_value = tl.broadcast_to(r135_value, [$(917504)])
    tl.store($(candidate_vadd_13_ptr) + r135_z, r135_value)
    z = tl.arange($(917504))
    value = tl.load($(candidate_vadd_13_ptr) + ((((((((z / $(256)) % $(3584)) * $(256)) + (z % $(256))) / $(114688)) % $(8)) * $(114688)) + ((((z / $(256)) % $(3584)) % $(448)) * $(256)) + (z % $(256))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(917504)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_nonabsorb_varlen_wgrad
