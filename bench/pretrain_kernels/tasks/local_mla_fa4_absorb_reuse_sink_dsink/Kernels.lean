import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_reuse_sink_dsink
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
  let parameter7_ptr : RegionName := "parameter7"
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
  let vmul_7_ptr : RegionName := "vmul_7"
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vsub_ptr : RegionName := "vsub"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_ptr : RegionName := "vcat"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vsum_1_ptr : RegionName := "vsum_1"
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
    r16_z = tl.arange($(8))
    r16_value = (tl.load($(parameter7_ptr) + r16_z, dtype=tl.float32) * 0.5)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(8)])
    tl.store($(vmul_7_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(16384))
    r17_value = tl.where((r17_z % $(128)) >= ((r17_z / $(128)) % $(128)) + $(1), true, false)
    r17_value = tl.broadcast_to(r17_value, [$(16384)])
    tl.store($(vtriu_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = tl.where(tl.load($(vtriu_ptr) + ((((r18_z / $(128)) % $(128)) * $(128)) + (r18_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(vmasked_fill_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(8))
    r19_value = tl.load($(vmul_7_ptr) + r19_z, dtype=tl.float32)
    r19_value = r19_value.to(tl.bfloat16)
    r19_value = tl.broadcast_to(r19_value, [$(8)])
    tl.store($(vto_copy_18_ptr) + r19_z, r19_value)
    r20_row = tl.arange($(256))
    r20_col = tl.arange($(256))
    r20_lane = tl.arange($(512))
    r20_left = tl.load($(input0_ptr) + ((((((r20_row[:, None] * $(512)) + r20_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r20_row[:, None] * $(512)) + r20_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r20_row[:, None] * $(512)) + r20_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r20_right = tl.load($(vto_copy_ptr) + (((r20_col[None, :] % $(256)) * $(512)) + (r20_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r20_value = tl.dot(r20_left, r20_right)
    r20_value = r20_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r20_row[:, None] * $(256)) + r20_col[None, :]), r20_value)
    r21_z = tl.arange($(65536))
    r21_value = tl.load($(vmm_ptr) + ((((r21_z / $(32768)) % $(2)) * $(32768)) + (((r21_z / $(256)) % $(128)) * $(256)) + (r21_z % $(256))), dtype=tl.bfloat16)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(65536)])
    tl.store($(vto_copy_19_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(65536))
    r22_value = ((tl.load($(vto_copy_19_ptr) + ((((r22_z / $(32768)) % $(2)) * $(32768)) + (((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_19_ptr) + ((((r22_z / $(32768)) % $(2)) * $(32768)) + (((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32)))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(65536)])
    tl.store($(vpow_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(256))
    r23_lane = tl.arange($(256))
    r23_value = (tl.sum(tl.load($(vpow_1_ptr) + ((((r23_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r23_z[:, None] % $(128)) * $(256)) + r23_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(256)])
    tl.store($(vmean_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(256))
    r24_value = (tl.load($(vmean_ptr) + ((((r24_z / $(128)) % $(2)) * $(128)) + (r24_z % $(128))), dtype=tl.float32) + 0.000001)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(256)])
    tl.store($(vadd_2_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(256))
    r25_value = tl.rsqrt(tl.load($(vadd_2_ptr) + ((((r25_z / $(128)) % $(2)) * $(128)) + (r25_z % $(128))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(256)])
    tl.store($(vrsqrt_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(65536))
    r26_value = (tl.load($(vto_copy_19_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(256)) % $(128)) * $(256)) + (r26_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r26_z / $(32768)) % $(2)) * $(128)) + ((r26_z / $(256)) % $(128))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(65536)])
    tl.store($(vmul_8_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(65536))
    r27_value = (tl.cast(tl.load($(vto_copy_1_ptr) + (r27_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_8_ptr) + ((((r27_z / $(32768)) % $(2)) * $(32768)) + (((r27_z / $(256)) % $(128)) * $(256)) + (r27_z % $(256))), dtype=tl.float32))
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(65536)])
    tl.store($(vmul_9_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(65536))
    r28_value = tl.load($(vmul_9_ptr) + ((((r28_z / $(32768)) % $(2)) * $(32768)) + (((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(65536)])
    tl.store($(vto_copy_20_ptr) + r28_z, r28_value)
    r29_row = tl.arange($(256))
    r29_col = tl.arange($(1536))
    r29_lane = tl.arange($(256))
    r29_left = tl.load($(vto_copy_20_ptr) + ((((((r29_row[:, None] * $(256)) + r29_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r29_row[:, None] * $(256)) + r29_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r29_row[:, None] * $(256)) + r29_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r29_right = tl.load($(vto_copy_2_ptr) + (((r29_col[None, :] % $(1536)) * $(256)) + (r29_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r29_value = tl.dot(r29_left, r29_right)
    r29_value = r29_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r29_row[:, None] * $(1536)) + r29_col[None, :]), r29_value)
    r30_row = tl.arange($(256))
    r30_col = tl.arange($(576))
    r30_lane = tl.arange($(512))
    r30_left = tl.load($(input0_ptr) + ((((((r30_row[:, None] * $(512)) + r30_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r30_row[:, None] * $(512)) + r30_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r30_row[:, None] * $(512)) + r30_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r30_right = tl.load($(vto_copy_3_ptr) + (((r30_col[None, :] % $(576)) * $(512)) + (r30_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r30_value = tl.dot(r30_left, r30_right)
    r30_value = r30_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r30_row[:, None] * $(576)) + r30_col[None, :]), r30_value)
    r31_z = tl.arange($(131072))
    r31_value = tl.load($(vmm_2_ptr) + ((((r31_z / $(65536)) % $(2)) * $(73728)) + (((r31_z / $(512)) % $(128)) * $(576)) + (r31_z % $(512))), dtype=tl.bfloat16)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(131072)])
    tl.store($(vto_copy_21_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(131072))
    r32_value = ((tl.load($(vto_copy_21_ptr) + ((((r32_z / $(65536)) % $(2)) * $(65536)) + (((r32_z / $(512)) % $(128)) * $(512)) + (r32_z % $(512))), dtype=tl.float32)) * (tl.load($(vto_copy_21_ptr) + ((((r32_z / $(65536)) % $(2)) * $(65536)) + (((r32_z / $(512)) % $(128)) * $(512)) + (r32_z % $(512))), dtype=tl.float32)))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(131072)])
    tl.store($(vpow_2_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(256))
    r33_lane = tl.arange($(512))
    r33_value = (tl.sum(tl.load($(vpow_2_ptr) + ((((r33_z[:, None] / $(128)) % $(2)) * $(65536)) + ((r33_z[:, None] % $(128)) * $(512)) + r33_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(256)])
    tl.store($(vmean_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(256))
    r34_value = (tl.load($(vmean_1_ptr) + ((((r34_z / $(128)) % $(2)) * $(128)) + (r34_z % $(128))), dtype=tl.float32) + 0.000001)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(256)])
    tl.store($(vadd_3_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(256))
    r35_value = tl.rsqrt(tl.load($(vadd_3_ptr) + ((((r35_z / $(128)) % $(2)) * $(128)) + (r35_z % $(128))), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(256)])
    tl.store($(vrsqrt_1_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(131072))
    r36_value = (tl.load($(vto_copy_21_ptr) + ((((r36_z / $(65536)) % $(2)) * $(65536)) + (((r36_z / $(512)) % $(128)) * $(512)) + (r36_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((((r36_z / $(65536)) % $(2)) * $(128)) + ((r36_z / $(512)) % $(128))), dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(131072)])
    tl.store($(vmul_10_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(131072))
    r37_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r37_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_10_ptr) + ((((r37_z / $(65536)) % $(2)) * $(65536)) + (((r37_z / $(512)) % $(128)) * $(512)) + (r37_z % $(512))), dtype=tl.float32))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(131072)])
    tl.store($(vmul_11_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(131072))
    r38_value = tl.load($(vmul_11_ptr) + ((((r38_z / $(65536)) % $(2)) * $(65536)) + (((r38_z / $(512)) % $(128)) * $(512)) + (r38_z % $(512))), dtype=tl.float32)
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(131072)])
    tl.store($(vto_copy_22_ptr) + r38_z, r38_value)
    r39_row = tl.arange($(256))
    r39_col = tl.arange($(2048))
    r39_lane = tl.arange($(512))
    r39_left = tl.load($(vto_copy_22_ptr) + ((((((r39_row[:, None] * $(512)) + r39_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r39_row[:, None] * $(512)) + r39_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r39_row[:, None] * $(512)) + r39_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r39_right = tl.load($(vto_copy_5_ptr) + (((r39_col[None, :] % $(2048)) * $(512)) + (r39_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r39_value = tl.dot(r39_left, r39_right)
    r39_value = r39_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r39_row[:, None] * $(2048)) + r39_col[None, :]), r39_value)
    r40_z = tl.arange($(65536))
    r40_value = (tl.load($(vmm_1_ptr) + ((((r40_z / $(32768)) % $(2)) * $(196608)) + (((r40_z / $(32)) % $(128)) * $(1536)) + (((r40_z / $(4096)) % $(8)) * $(192)) + ((r40_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(vmul_12_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(65536))
    r41_value = (tl.load($(vmm_1_ptr) + ((((r41_z / $(32768)) % $(2)) * $(196608)) + (((r41_z / $(32)) % $(128)) * $(1536)) + (((r41_z / $(4096)) % $(8)) * $(192)) + ((r41_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(65536)])
    tl.store($(vmul_13_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(65536))
    r42_value = (tl.load($(vmul_12_ptr) + ((((r42_z / $(32768)) % $(2)) * $(32768)) + (((r42_z / $(4096)) % $(8)) * $(4096)) + (((r42_z / $(32)) % $(128)) * $(32)) + (r42_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_13_ptr) + ((((r42_z / $(32768)) % $(2)) * $(32768)) + (((r42_z / $(4096)) % $(8)) * $(4096)) + (((r42_z / $(32)) % $(128)) * $(32)) + (r42_z % $(32))), dtype=tl.bfloat16))
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(65536)])
    tl.store($(vsub_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(65536))
    r43_value = (tl.load($(vmm_1_ptr) + ((((r43_z / $(32768)) % $(2)) * $(196608)) + (((r43_z / $(32)) % $(128)) * $(1536)) + (((r43_z / $(4096)) % $(8)) * $(192)) + ((r43_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(65536)])
    tl.store($(vmul_14_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(65536))
    r44_value = (tl.load($(vmm_1_ptr) + ((((r44_z / $(32768)) % $(2)) * $(196608)) + (((r44_z / $(32)) % $(128)) * $(1536)) + (((r44_z / $(4096)) % $(8)) * $(192)) + ((r44_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(65536)])
    tl.store($(vmul_15_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(65536))
    r45_value = (tl.load($(vmul_14_ptr) + ((((r45_z / $(32768)) % $(2)) * $(32768)) + (((r45_z / $(4096)) % $(8)) * $(4096)) + (((r45_z / $(32)) % $(128)) * $(32)) + (r45_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_15_ptr) + ((((r45_z / $(32768)) % $(2)) * $(32768)) + (((r45_z / $(4096)) % $(8)) * $(4096)) + (((r45_z / $(32)) % $(128)) * $(32)) + (r45_z % $(32))), dtype=tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(65536)])
    tl.store($(vadd_4_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(131072))
    r46_value = tl.where((r46_z % $(64)) < $(32), tl.load($(vsub_ptr) + ((((r46_z / $(65536)) % $(2)) * $(32768)) + (((r46_z / $(8192)) % $(8)) * $(4096)) + (((r46_z / $(64)) % $(128)) * $(32)) + (((r46_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_4_ptr) + ((((r46_z / $(65536)) % $(2)) * $(32768)) + (((r46_z / $(8192)) % $(8)) * $(4096)) + (((r46_z / $(64)) % $(128)) * $(32)) + (((r46_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(131072)])
    tl.store($(vcat_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(8192))
    r47_value = (tl.load($(vmm_2_ptr) + ((((r47_z / $(4096)) % $(2)) * $(73728)) + (((r47_z / $(32)) % $(128)) * $(576)) + ((r47_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(8192)])
    tl.store($(vmul_16_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(8192))
    r48_value = (tl.load($(vmm_2_ptr) + ((((r48_z / $(4096)) % $(2)) * $(73728)) + (((r48_z / $(32)) % $(128)) * $(576)) + ((r48_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(8192)])
    tl.store($(vmul_17_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(8192))
    r49_value = (tl.load($(vmul_16_ptr) + ((((r49_z / $(4096)) % $(2)) * $(4096)) + (((r49_z / $(32)) % $(128)) * $(32)) + (r49_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_17_ptr) + ((((r49_z / $(4096)) % $(2)) * $(4096)) + (((r49_z / $(32)) % $(128)) * $(32)) + (r49_z % $(32))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(8192)])
    tl.store($(vsub_1_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(8192))
    r50_value = (tl.load($(vmm_2_ptr) + ((((r50_z / $(4096)) % $(2)) * $(73728)) + (((r50_z / $(32)) % $(128)) * $(576)) + ((r50_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(8192)])
    tl.store($(vmul_18_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(8192))
    r51_value = (tl.load($(vmm_2_ptr) + ((((r51_z / $(4096)) % $(2)) * $(73728)) + (((r51_z / $(32)) % $(128)) * $(576)) + ((r51_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(8192)])
    tl.store($(vmul_19_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(8192))
    r52_value = (tl.load($(vmul_18_ptr) + ((((r52_z / $(4096)) % $(2)) * $(4096)) + (((r52_z / $(32)) % $(128)) * $(32)) + (r52_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_19_ptr) + ((((r52_z / $(4096)) % $(2)) * $(4096)) + (((r52_z / $(32)) % $(128)) * $(32)) + (r52_z % $(32))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(8192)])
    tl.store($(vadd_5_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(16384))
    r53_value = tl.where((r53_z % $(64)) < $(32), tl.load($(vsub_1_ptr) + ((((r53_z / $(8192)) % $(2)) * $(4096)) + (((r53_z / $(64)) % $(128)) * $(32)) + (((r53_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r53_z / $(8192)) % $(2)) * $(4096)) + (((r53_z / $(64)) % $(128)) * $(32)) + (((r53_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(16384)])
    tl.store($(vcat_1_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(393216))
    r54_value = tl.where((r54_z % $(192)) < $(128), tl.load($(vmm_1_ptr) + ((((((((r54_z / $(196608)) % $(2)) * $(196608)) + (((r54_z / $(192)) % $(128)) * $(1536)) + (((r54_z / $(24576)) % $(8)) * $(192)) + ((r54_z % $(192)) - $(0))) / $(196608)) % $(2)) * $(196608)) + (((((((r54_z / $(196608)) % $(2)) * $(196608)) + (((r54_z / $(192)) % $(128)) * $(1536)) + (((r54_z / $(24576)) % $(8)) * $(192)) + ((r54_z % $(192)) - $(0))) / $(1536)) % $(128)) * $(1536)) + (((((r54_z / $(196608)) % $(2)) * $(196608)) + (((r54_z / $(192)) % $(128)) * $(1536)) + (((r54_z / $(24576)) % $(8)) * $(192)) + ((r54_z % $(192)) - $(0))) % $(1536))), dtype=tl.bfloat16), tl.load($(vcat_ptr) + ((((r54_z / $(196608)) % $(2)) * $(65536)) + (((r54_z / $(24576)) % $(8)) * $(8192)) + (((r54_z / $(192)) % $(128)) * $(64)) + (((r54_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(393216)])
    tl.store($(vcat_2_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(393216))
    r55_value = tl.where((r55_z % $(192)) < $(128), tl.load($(vmm_3_ptr) + ((((((((r55_z / $(196608)) % $(2)) * $(262144)) + (((r55_z / $(192)) % $(128)) * $(2048)) + (((r55_z / $(24576)) % $(8)) * $(256)) + ((r55_z % $(192)) - $(0))) / $(262144)) % $(2)) * $(262144)) + (((((((r55_z / $(196608)) % $(2)) * $(262144)) + (((r55_z / $(192)) % $(128)) * $(2048)) + (((r55_z / $(24576)) % $(8)) * $(256)) + ((r55_z % $(192)) - $(0))) / $(2048)) % $(128)) * $(2048)) + (((((r55_z / $(196608)) % $(2)) * $(262144)) + (((r55_z / $(192)) % $(128)) * $(2048)) + (((r55_z / $(24576)) % $(8)) * $(256)) + ((r55_z % $(192)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_1_ptr) + ((((r55_z / $(196608)) % $(2)) * $(8192)) + (((r55_z / $(192)) % $(128)) * $(64)) + (((r55_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(393216)])
    tl.store($(vcat_3_ptr) + r55_z, r55_value)
    r56_row = tl.arange($(128))
    r56_col = tl.arange($(128))
    r56_lane = tl.arange($(192))
    for r56_group in range($(16)) {
      r56_left = tl.load($(vcat_2_ptr) + (((((((r56_group % $(16)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r56_group % $(16)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r56_group % $(16)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r56_group % $(16)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r56_right = tl.load($(vcat_3_ptr) + (((((((r56_group % $(16)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r56_group % $(16)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r56_group % $(16)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) % $(128)) * $(192)) + (((((r56_group % $(16)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r56_value = tl.dot(r56_left, r56_right)
      r56_value = r56_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r56_group * $(16384)) + (r56_row[:, None] * $(128)) + r56_col[None, :]), r56_value)
    }
    r57_z = tl.arange($(262144))
    r57_value = (tl.load($(vbmm_ptr) + ((((r57_z / $(131072)) % $(2)) * $(131072)) + (((r57_z / $(16384)) % $(8)) * $(16384)) + (((r57_z / $(128)) % $(128)) * $(128)) + (r57_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(262144)])
    tl.store($(vmul_20_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(262144))
    r58_value = (tl.load($(vmul_20_ptr) + ((((r58_z / $(131072)) % $(2)) * $(131072)) + (((r58_z / $(16384)) % $(8)) * $(16384)) + (((r58_z / $(128)) % $(128)) * $(128)) + (r58_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r58_z / $(131072)) % $(2)) * $(16384)) + (((r58_z / $(128)) % $(128)) * $(128)) + (r58_z % $(128))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(262144)])
    tl.store($(vadd_6_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(264192))
    r59_value = tl.where((r59_z % $(129)) < $(128), tl.load($(vadd_6_ptr) + ((((r59_z / $(132096)) % $(2)) * $(131072)) + (((r59_z / $(16512)) % $(8)) * $(16384)) + (((r59_z / $(129)) % $(128)) * $(128)) + (((r59_z % $(129)) - $(0)) % $(128))), dtype=tl.bfloat16), tl.load($(vto_copy_18_ptr) + ((r59_z / $(16512)) % $(8)), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(264192)])
    tl.store($(vcat_4_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(264192))
    r60_value = tl.load($(vcat_4_ptr) + ((((r60_z / $(132096)) % $(2)) * $(132096)) + (((r60_z / $(16512)) % $(8)) * $(16512)) + (((r60_z / $(129)) % $(128)) * $(129)) + (r60_z % $(129))), dtype=tl.bfloat16)
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(264192)])
    tl.store($(vto_copy_23_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(264192))
    r61_lane = tl.arange($(129))
    r61_row_values = tl.load($(vto_copy_23_ptr) + ((((r61_z[:, None] / $(132096)) % $(2)) * $(132096)) + (((r61_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r61_z[:, None] / $(129)) % $(128)) * $(129)) + (r61_lane[None, :] % $(129))), dtype=tl.float32)
    r61_maximum = tl.max(r61_row_values, axis=1, keep_dims=true)
    r61_weights = tl.exp(r61_row_values - r61_maximum)
    r61_total = tl.sum(r61_weights, axis=1)
    r61_maximum = tl.max(r61_row_values, axis=1)
    r61_value = tl.exp(tl.load($(vto_copy_23_ptr) + ((((r61_z / $(132096)) % $(2)) * $(132096)) + (((r61_z / $(16512)) % $(8)) * $(16512)) + (((r61_z / $(129)) % $(128)) * $(129)) + (r61_z % $(129))), dtype=tl.float32) - r61_maximum) / r61_total
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(264192)])
    tl.store($(vsoftmax_ptr) + r61_z, r61_value)
    r62_row = tl.arange($(256))
    r62_col = tl.arange($(1024))
    r62_lane = tl.arange($(512))
    r62_left = tl.load($(input1_ptr) + ((((((r62_row[:, None] * $(512)) + r62_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r62_row[:, None] * $(512)) + r62_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r62_row[:, None] * $(512)) + r62_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r62_right = tl.load($(vto_copy_6_ptr) + (((r62_lane[:, None] % $(512)) * $(1024)) + (r62_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r62_value = tl.dot(r62_left, r62_right)
    r62_value = r62_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r62_row[:, None] * $(1024)) + r62_col[None, :]), r62_value)
    r63_row = tl.arange($(128))
    r63_col = tl.arange($(128))
    r63_lane = tl.arange($(128))
    for r63_group in range($(16)) {
      r63_left = tl.load($(vmm_5_ptr) + (((((((r63_group % $(16)) * $(16384)) + (r63_row[:, None] * $(128)) + r63_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r63_group % $(16)) * $(16384)) + (r63_row[:, None] * $(128)) + r63_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r63_group % $(16)) * $(16384)) + (r63_row[:, None] * $(128)) + r63_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r63_group % $(16)) * $(16384)) + (r63_row[:, None] * $(128)) + r63_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r63_right = tl.load($(vmm_3_ptr) + (((((((r63_group % $(16)) * $(16384)) + (r63_col[None, :] * $(128)) + r63_lane[:, None]) / $(131072)) % $(2)) * $(262144)) + ((((((r63_group % $(16)) * $(16384)) + (r63_col[None, :] * $(128)) + r63_lane[:, None]) / $(128)) % $(128)) * $(2048)) + ((((((r63_group % $(16)) * $(16384)) + (r63_col[None, :] * $(128)) + r63_lane[:, None]) / $(16384)) % $(8)) * $(256)) + ((((r63_group % $(16)) * $(16384)) + (r63_col[None, :] * $(128)) + r63_lane[:, None]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r63_value = tl.dot(r63_left, r63_right)
      r63_value = r63_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r63_group * $(16384)) + (r63_row[:, None] * $(128)) + r63_col[None, :]), r63_value)
    }
    r64_z = tl.arange($(264192))
    r64_value = tl.where((((r64_z % $(129)) >= $(0)) & ((r64_z % $(129)) < $(128)) & (((r64_z % $(129)) - $(0)) % $(1) == $(0))), tl.load($(vbmm_2_ptr) + ((((((((r64_z / $(132096)) % $(2)) * $(131072)) + (((r64_z / $(16512)) % $(8)) * $(16384)) + (((r64_z / $(129)) % $(128)) * $(128)) + ((r64_z % $(129)) - $(0))) / $(16384)) % $(16)) * $(16384)) + (((((((r64_z / $(132096)) % $(2)) * $(131072)) + (((r64_z / $(16512)) % $(8)) * $(16384)) + (((r64_z / $(129)) % $(128)) * $(128)) + ((r64_z % $(129)) - $(0))) / $(128)) % $(128)) * $(128)) + (((((r64_z / $(132096)) % $(2)) * $(131072)) + (((r64_z / $(16512)) % $(8)) * $(16384)) + (((r64_z / $(129)) % $(128)) * $(128)) + ((r64_z % $(129)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(264192)])
    tl.store($(vslice_backward_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(264192))
    r65_value = tl.load($(vslice_backward_ptr) + ((((r65_z / $(132096)) % $(2)) * $(132096)) + (((r65_z / $(16512)) % $(8)) * $(16512)) + (((r65_z / $(129)) % $(128)) * $(129)) + (r65_z % $(129))), dtype=tl.bfloat16)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(264192)])
    tl.store($(vto_copy_25_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(264192))
    r66_lane = tl.arange($(129))
    r66_row_values = tl.load($(vsoftmax_ptr) + ((((r66_z[:, None] / $(132096)) % $(2)) * $(132096)) + (((r66_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r66_z[:, None] / $(129)) % $(128)) * $(129)) + (r66_lane[None, :] % $(129))), dtype=tl.float32)
    r66_row_grad = tl.load($(vto_copy_25_ptr) + ((((r66_z[:, None] / $(132096)) % $(2)) * $(132096)) + (((r66_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r66_z[:, None] / $(129)) % $(128)) * $(129)) + (r66_lane[None, :] % $(129))), dtype=tl.float32)
    r66_weighted = r66_row_grad * r66_row_values
    r66_dot_grad = tl.sum(r66_weighted, axis=1)
    r66_value = (tl.load($(vto_copy_25_ptr) + ((((r66_z / $(132096)) % $(2)) * $(132096)) + (((r66_z / $(16512)) % $(8)) * $(16512)) + (((r66_z / $(129)) % $(128)) * $(129)) + (r66_z % $(129))), dtype=tl.float32) - r66_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r66_z / $(132096)) % $(2)) * $(132096)) + (((r66_z / $(16512)) % $(8)) * $(16512)) + (((r66_z / $(129)) % $(128)) * $(129)) + (r66_z % $(129))), dtype=tl.float32)
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(264192)])
    tl.store($(vsoftmax_backward_data_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(264192))
    r67_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r67_z / $(132096)) % $(2)) * $(132096)) + (((r67_z / $(16512)) % $(8)) * $(16512)) + (((r67_z / $(129)) % $(128)) * $(129)) + (r67_z % $(129))), dtype=tl.float32)
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(264192)])
    tl.store($(vto_copy_26_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(8))
    r68_lane = tl.arange($(256))
    r68_value = tl.sum(tl.load($(vto_copy_26_ptr) + ((((r68_lane[None, :] / $(128)) % $(2)) * $(132096)) + (r68_z[:, None] * $(16512)) + ((r68_lane[None, :] % $(128)) * $(129)) + $(128)), dtype=tl.bfloat16), axis=1)
    r68_value = r68_value.to(tl.bfloat16)
    r68_value = tl.broadcast_to(r68_value, [$(8)])
    tl.store($(vsum_1_ptr) + r68_z, r68_value)
    z = tl.arange($(8))
    value = tl.load($(vsum_1_ptr) + z, dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(8)])
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
  let parameter7_ptr : RegionName := "parameter7"
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
  let vmul_7_ptr : RegionName := "vmul_7"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vsub_ptr : RegionName := "vsub"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_ptr : RegionName := "vcat"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vsum_1_ptr : RegionName := "vsum_1"
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
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vpow_2_ptr : RegionName := "candidate_vpow_2"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vcat_1_ptr : RegionName := "candidate_vcat_1"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_2_ptr : RegionName := "candidate_vsub_2"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vcat_2_ptr : RegionName := "candidate_vcat_2"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vlogsumexp_ptr : RegionName := "candidate_vlogsumexp"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_3_ptr : RegionName := "candidate_vsub_3"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vcat_3_ptr : RegionName := "candidate_vcat_3"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vlogsumexp_1_ptr : RegionName := "candidate_vlogsumexp_1"
  let candidate_vcat_4_ptr : RegionName := "candidate_vcat_4"
  let candidate_vcat_5_ptr : RegionName := "candidate_vcat_5"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vmm_8_ptr : RegionName := "candidate_vmm_8"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vsub_7_ptr : RegionName := "candidate_vsub_7"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vexp_3_ptr : RegionName := "candidate_vexp_3"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
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
    r16_z = tl.arange($(8))
    r16_value = (tl.load($(parameter7_ptr) + r16_z, dtype=tl.float32) * 0.5)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(8)])
    tl.store($(candidate_vmul_7_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(8))
    r17_value = tl.load($(candidate_vmul_7_ptr) + r17_z, dtype=tl.float32)
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(8)])
    tl.store($(candidate_vto_copy_18_ptr) + r17_z, r17_value)
    r18_row = tl.arange($(256))
    r18_col = tl.arange($(256))
    r18_lane = tl.arange($(512))
    r18_left = tl.load($(input0_ptr) + ((((((r18_row[:, None] * $(512)) + r18_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r18_row[:, None] * $(512)) + r18_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r18_row[:, None] * $(512)) + r18_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r18_right = tl.load($(candidate_vto_copy_ptr) + (((r18_col[None, :] % $(256)) * $(512)) + (r18_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r18_value = tl.dot(r18_left, r18_right)
    r18_value = r18_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r18_row[:, None] * $(256)) + r18_col[None, :]), r18_value)
    r19_z = tl.arange($(65536))
    r19_value = tl.load($(candidate_vmm_ptr) + ((((r19_z / $(32768)) % $(2)) * $(32768)) + (((r19_z / $(256)) % $(128)) * $(256)) + (r19_z % $(256))), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(65536)])
    tl.store($(candidate_vto_copy_19_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(65536))
    r20_value = ((tl.load($(candidate_vto_copy_19_ptr) + ((((r20_z / $(32768)) % $(2)) * $(32768)) + (((r20_z / $(256)) % $(128)) * $(256)) + (r20_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_19_ptr) + ((((r20_z / $(32768)) % $(2)) * $(32768)) + (((r20_z / $(256)) % $(128)) * $(256)) + (r20_z % $(256))), dtype=tl.float32)))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(65536)])
    tl.store($(candidate_vpow_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(256))
    r21_lane = tl.arange($(256))
    r21_value = (tl.sum(tl.load($(candidate_vpow_1_ptr) + ((((r21_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r21_z[:, None] % $(128)) * $(256)) + r21_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(256)])
    tl.store($(candidate_vmean_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(256))
    r22_value = (tl.load($(candidate_vmean_ptr) + ((((r22_z / $(128)) % $(2)) * $(128)) + (r22_z % $(128))), dtype=tl.float32) + 0.000001)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(256)])
    tl.store($(candidate_vadd_2_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(256))
    r23_value = tl.rsqrt(tl.load($(candidate_vadd_2_ptr) + ((((r23_z / $(128)) % $(2)) * $(128)) + (r23_z % $(128))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(256)])
    tl.store($(candidate_vrsqrt_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = (tl.load($(candidate_vto_copy_19_ptr) + ((((r24_z / $(32768)) % $(2)) * $(32768)) + (((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r24_z / $(32768)) % $(2)) * $(128)) + ((r24_z / $(256)) % $(128))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(candidate_vmul_8_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(65536))
    r25_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r25_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_8_ptr) + ((((r25_z / $(32768)) % $(2)) * $(32768)) + (((r25_z / $(256)) % $(128)) * $(256)) + (r25_z % $(256))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(65536)])
    tl.store($(candidate_vmul_9_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(65536))
    r26_value = tl.load($(candidate_vmul_9_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(256)) % $(128)) * $(256)) + (r26_z % $(256))), dtype=tl.float32)
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(65536)])
    tl.store($(candidate_vto_copy_20_ptr) + r26_z, r26_value)
    r27_row = tl.arange($(256))
    r27_col = tl.arange($(1536))
    r27_lane = tl.arange($(256))
    r27_left = tl.load($(candidate_vto_copy_20_ptr) + ((((((r27_row[:, None] * $(256)) + r27_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r27_row[:, None] * $(256)) + r27_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r27_row[:, None] * $(256)) + r27_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r27_right = tl.load($(candidate_vto_copy_2_ptr) + (((r27_col[None, :] % $(1536)) * $(256)) + (r27_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r27_value = tl.dot(r27_left, r27_right)
    r27_value = r27_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r27_row[:, None] * $(1536)) + r27_col[None, :]), r27_value)
    r28_row = tl.arange($(256))
    r28_col = tl.arange($(576))
    r28_lane = tl.arange($(512))
    r28_left = tl.load($(input0_ptr) + ((((((r28_row[:, None] * $(512)) + r28_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r28_row[:, None] * $(512)) + r28_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r28_row[:, None] * $(512)) + r28_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r28_right = tl.load($(candidate_vto_copy_3_ptr) + (((r28_col[None, :] % $(576)) * $(512)) + (r28_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r28_value = tl.dot(r28_left, r28_right)
    r28_value = r28_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r28_row[:, None] * $(576)) + r28_col[None, :]), r28_value)
    r29_z = tl.arange($(131072))
    r29_value = tl.load($(candidate_vmm_2_ptr) + ((((r29_z / $(65536)) % $(2)) * $(73728)) + (((r29_z / $(512)) % $(128)) * $(576)) + (r29_z % $(512))), dtype=tl.bfloat16)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(131072)])
    tl.store($(candidate_vto_copy_21_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(131072))
    r30_value = ((tl.load($(candidate_vto_copy_21_ptr) + ((((r30_z / $(65536)) % $(2)) * $(65536)) + (((r30_z / $(512)) % $(128)) * $(512)) + (r30_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_21_ptr) + ((((r30_z / $(65536)) % $(2)) * $(65536)) + (((r30_z / $(512)) % $(128)) * $(512)) + (r30_z % $(512))), dtype=tl.float32)))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(131072)])
    tl.store($(candidate_vpow_2_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(256))
    r31_lane = tl.arange($(512))
    r31_value = (tl.sum(tl.load($(candidate_vpow_2_ptr) + ((((r31_z[:, None] / $(128)) % $(2)) * $(65536)) + ((r31_z[:, None] % $(128)) * $(512)) + r31_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(256)])
    tl.store($(candidate_vmean_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(256))
    r32_value = (tl.load($(candidate_vmean_1_ptr) + ((((r32_z / $(128)) % $(2)) * $(128)) + (r32_z % $(128))), dtype=tl.float32) + 0.000001)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(256)])
    tl.store($(candidate_vadd_3_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(256))
    r33_value = tl.rsqrt(tl.load($(candidate_vadd_3_ptr) + ((((r33_z / $(128)) % $(2)) * $(128)) + (r33_z % $(128))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(256)])
    tl.store($(candidate_vrsqrt_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(131072))
    r34_value = (tl.load($(candidate_vto_copy_21_ptr) + ((((r34_z / $(65536)) % $(2)) * $(65536)) + (((r34_z / $(512)) % $(128)) * $(512)) + (r34_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((((r34_z / $(65536)) % $(2)) * $(128)) + ((r34_z / $(512)) % $(128))), dtype=tl.float32))
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(131072)])
    tl.store($(candidate_vmul_10_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(131072))
    r35_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r35_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_10_ptr) + ((((r35_z / $(65536)) % $(2)) * $(65536)) + (((r35_z / $(512)) % $(128)) * $(512)) + (r35_z % $(512))), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(131072)])
    tl.store($(candidate_vmul_11_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(131072))
    r36_value = tl.load($(candidate_vmul_11_ptr) + ((((r36_z / $(65536)) % $(2)) * $(65536)) + (((r36_z / $(512)) % $(128)) * $(512)) + (r36_z % $(512))), dtype=tl.float32)
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(131072)])
    tl.store($(candidate_vto_copy_22_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(65536))
    r37_value = (tl.load($(candidate_vmm_1_ptr) + ((((r37_z / $(32768)) % $(2)) * $(196608)) + (((r37_z / $(32)) % $(128)) * $(1536)) + (((r37_z / $(4096)) % $(8)) * $(192)) + ((r37_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(65536)])
    tl.store($(candidate_vmul_12_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(65536))
    r38_value = (tl.load($(candidate_vmm_1_ptr) + ((((r38_z / $(32768)) % $(2)) * $(196608)) + (((r38_z / $(32)) % $(128)) * $(1536)) + (((r38_z / $(4096)) % $(8)) * $(192)) + ((r38_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(65536)])
    tl.store($(candidate_vmul_13_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(65536))
    r39_value = (tl.load($(candidate_vmul_12_ptr) + ((((r39_z / $(32768)) % $(2)) * $(32768)) + (((r39_z / $(4096)) % $(8)) * $(4096)) + (((r39_z / $(32)) % $(128)) * $(32)) + (r39_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_13_ptr) + ((((r39_z / $(32768)) % $(2)) * $(32768)) + (((r39_z / $(4096)) % $(8)) * $(4096)) + (((r39_z / $(32)) % $(128)) * $(32)) + (r39_z % $(32))), dtype=tl.bfloat16))
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(65536)])
    tl.store($(candidate_vsub_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(65536))
    r40_value = (tl.load($(candidate_vmm_1_ptr) + ((((r40_z / $(32768)) % $(2)) * $(196608)) + (((r40_z / $(32)) % $(128)) * $(1536)) + (((r40_z / $(4096)) % $(8)) * $(192)) + ((r40_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(candidate_vmul_14_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(65536))
    r41_value = (tl.load($(candidate_vmm_1_ptr) + ((((r41_z / $(32768)) % $(2)) * $(196608)) + (((r41_z / $(32)) % $(128)) * $(1536)) + (((r41_z / $(4096)) % $(8)) * $(192)) + ((r41_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(65536)])
    tl.store($(candidate_vmul_15_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(65536))
    r42_value = (tl.load($(candidate_vmul_14_ptr) + ((((r42_z / $(32768)) % $(2)) * $(32768)) + (((r42_z / $(4096)) % $(8)) * $(4096)) + (((r42_z / $(32)) % $(128)) * $(32)) + (r42_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_15_ptr) + ((((r42_z / $(32768)) % $(2)) * $(32768)) + (((r42_z / $(4096)) % $(8)) * $(4096)) + (((r42_z / $(32)) % $(128)) * $(32)) + (r42_z % $(32))), dtype=tl.bfloat16))
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(65536)])
    tl.store($(candidate_vadd_4_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(131072))
    r43_value = tl.where((r43_z % $(64)) < $(32), tl.load($(candidate_vsub_ptr) + ((((r43_z / $(65536)) % $(2)) * $(32768)) + (((r43_z / $(8192)) % $(8)) * $(4096)) + (((r43_z / $(64)) % $(128)) * $(32)) + (((r43_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_4_ptr) + ((((r43_z / $(65536)) % $(2)) * $(32768)) + (((r43_z / $(8192)) % $(8)) * $(4096)) + (((r43_z / $(64)) % $(128)) * $(32)) + (((r43_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(131072)])
    tl.store($(candidate_vcat_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(8192))
    r44_value = (tl.load($(candidate_vmm_2_ptr) + ((((r44_z / $(4096)) % $(2)) * $(73728)) + (((r44_z / $(32)) % $(128)) * $(576)) + ((r44_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(8192)])
    tl.store($(candidate_vmul_16_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(8192))
    r45_value = (tl.load($(candidate_vmm_2_ptr) + ((((r45_z / $(4096)) % $(2)) * $(73728)) + (((r45_z / $(32)) % $(128)) * $(576)) + ((r45_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(8192)])
    tl.store($(candidate_vmul_17_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(8192))
    r46_value = (tl.load($(candidate_vmul_16_ptr) + ((((r46_z / $(4096)) % $(2)) * $(4096)) + (((r46_z / $(32)) % $(128)) * $(32)) + (r46_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_17_ptr) + ((((r46_z / $(4096)) % $(2)) * $(4096)) + (((r46_z / $(32)) % $(128)) * $(32)) + (r46_z % $(32))), dtype=tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(8192)])
    tl.store($(candidate_vsub_1_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(8192))
    r47_value = (tl.load($(candidate_vmm_2_ptr) + ((((r47_z / $(4096)) % $(2)) * $(73728)) + (((r47_z / $(32)) % $(128)) * $(576)) + ((r47_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(8192)])
    tl.store($(candidate_vmul_18_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(8192))
    r48_value = (tl.load($(candidate_vmm_2_ptr) + ((((r48_z / $(4096)) % $(2)) * $(73728)) + (((r48_z / $(32)) % $(128)) * $(576)) + ((r48_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(8192)])
    tl.store($(candidate_vmul_19_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(8192))
    r49_value = (tl.load($(candidate_vmul_18_ptr) + ((((r49_z / $(4096)) % $(2)) * $(4096)) + (((r49_z / $(32)) % $(128)) * $(32)) + (r49_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_19_ptr) + ((((r49_z / $(4096)) % $(2)) * $(4096)) + (((r49_z / $(32)) % $(128)) * $(32)) + (r49_z % $(32))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(8192)])
    tl.store($(candidate_vadd_5_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(16384))
    r50_value = tl.where((r50_z % $(64)) < $(32), tl.load($(candidate_vsub_1_ptr) + ((((r50_z / $(8192)) % $(2)) * $(4096)) + (((r50_z / $(64)) % $(128)) * $(32)) + (((r50_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_5_ptr) + ((((r50_z / $(8192)) % $(2)) * $(4096)) + (((r50_z / $(64)) % $(128)) * $(32)) + (((r50_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(16384)])
    tl.store($(candidate_vcat_1_ptr) + r50_z, r50_value)
    r51_row = tl.arange($(256))
    r51_col = tl.arange($(512))
    r51_lane = tl.arange($(128))
    for r51_group in range($(8)) {
      r51_left = tl.load($(candidate_vmm_1_ptr) + (((((((((((r51_group % $(8)) * $(32768)) + (r51_row[:, None] * $(128)) + r51_lane[None, :]) / $(128)) % $(256)) * $(1024)) + ((((((r51_group % $(8)) * $(32768)) + (r51_row[:, None] * $(128)) + r51_lane[None, :]) / $(32768)) % $(8)) * $(128)) + ((((r51_group % $(8)) * $(32768)) + (r51_row[:, None] * $(128)) + r51_lane[None, :]) % $(128))) / $(131072)) % $(2)) * $(196608)) + (((((((r51_group % $(8)) * $(32768)) + (r51_row[:, None] * $(128)) + r51_lane[None, :]) / $(128)) % $(256)) % $(128)) * $(1536)) + ((((((r51_group % $(8)) * $(32768)) + (r51_row[:, None] * $(128)) + r51_lane[None, :]) / $(32768)) % $(8)) * $(192)) + ((((r51_group % $(8)) * $(32768)) + (r51_row[:, None] * $(128)) + r51_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r51_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r51_group % $(8)) * $(65536)) + (r51_lane[:, None] * $(512)) + r51_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r51_group % $(8)) * $(65536)) + (r51_lane[:, None] * $(512)) + r51_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r51_group % $(8)) * $(65536)) + (r51_lane[:, None] * $(512)) + r51_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r51_value = tl.dot(r51_left, r51_right)
      r51_value = r51_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r51_group * $(131072)) + (r51_row[:, None] * $(512)) + r51_col[None, :]), r51_value)
    }
    r52_z = tl.arange($(65536))
    r52_value = tl.load($(candidate_vcat_ptr) + ((((r52_z / $(8192)) % $(8)) * $(8192)) + (((r52_z / $(64)) % $(128)) * $(64)) + (r52_z % $(64))), dtype=tl.bfloat16)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(65536)])
    tl.store($(candidate_vto_copy_23_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(8192))
    r53_value = tl.load($(candidate_vcat_1_ptr) + ((((r53_z / $(64)) % $(128)) * $(64)) + (r53_z % $(64))), dtype=tl.bfloat16)
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(8192)])
    tl.store($(candidate_vto_copy_24_ptr) + r53_z, r53_value)
    r54_row = tl.arange($(128))
    r54_col = tl.arange($(128))
    r54_lane = tl.arange($(64))
    for r54_group in range($(8)) {
      r54_left = tl.load($(candidate_vto_copy_23_ptr) + (((((((r54_group % $(8)) * $(8192)) + (r54_row[:, None] * $(64)) + r54_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r54_group % $(8)) * $(8192)) + (r54_row[:, None] * $(64)) + r54_lane[None, :]) / $(64)) % $(128)) * $(64)) + ((((r54_group % $(8)) * $(8192)) + (r54_row[:, None] * $(64)) + r54_lane[None, :]) % $(64))), dtype=tl.float32)
      r54_right = tl.load($(candidate_vto_copy_24_ptr) + ((((((r54_group % $(8)) * $(8192)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) % $(128)) * $(64)) + (((((r54_group % $(8)) * $(8192)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) / $(128)) % $(64))), dtype=tl.float32)
      r54_value = tl.dot(r54_left, r54_right)
      r54_value = r54_value.to(tl.float32)
      tl.store($(candidate_vbmm_1_ptr) + ((r54_group * $(16384)) + (r54_row[:, None] * $(128)) + r54_col[None, :]), r54_value)
    }
    r55_z = tl.arange($(524288))
    r55_value = tl.load($(candidate_vbmm_ptr) + ((((r55_z / $(65536)) % $(8)) * $(131072)) + (((r55_z / $(512)) % $(128)) * $(512)) + (r55_z % $(512))), dtype=tl.bfloat16)
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(524288)])
    tl.store($(candidate_vto_copy_25_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(65536))
    r56_value = tl.load($(candidate_vto_copy_22_ptr) + ((((r56_z / $(512)) % $(128)) * $(512)) + (r56_z % $(512))), dtype=tl.bfloat16)
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(65536)])
    tl.store($(candidate_vto_copy_26_ptr) + r56_z, r56_value)
    r57_row = tl.arange($(1024))
    r57_col = tl.arange($(128))
    r57_lane = tl.arange($(512))
    r57_left = tl.load($(candidate_vto_copy_25_ptr) + ((((((r57_row[:, None] * $(512)) + r57_lane[None, :]) / $(65536)) % $(8)) * $(65536)) + (((((r57_row[:, None] * $(512)) + r57_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r57_row[:, None] * $(512)) + r57_lane[None, :]) % $(512))), dtype=tl.float32)
    r57_right = tl.load($(candidate_vto_copy_26_ptr) + (((((r57_lane[:, None] * $(128)) + r57_col[None, :]) % $(128)) * $(512)) + ((((r57_lane[:, None] * $(128)) + r57_col[None, :]) / $(128)) % $(512))), dtype=tl.float32)
    r57_value = tl.dot(r57_left, r57_right)
    r57_value = r57_value.to(tl.float32)
    tl.store($(candidate_vmm_3_ptr) + ((r57_row[:, None] * $(128)) + r57_col[None, :]), r57_value)
    r58_z = tl.arange($(131072))
    r58_value = (tl.load($(candidate_vbmm_1_ptr) + ((((r58_z / $(16384)) % $(8)) * $(16384)) + (((r58_z / $(128)) % $(128)) * $(128)) + (r58_z % $(128))), dtype=tl.float32) + tl.load($(candidate_vmm_3_ptr) + ((((r58_z / $(16384)) % $(8)) * $(16384)) + (((r58_z / $(128)) % $(128)) * $(128)) + (r58_z % $(128))), dtype=tl.float32))
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(131072)])
    tl.store($(candidate_vadd_6_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(131072))
    r59_value = (tl.load($(candidate_vadd_6_ptr) + ((((r59_z / $(16384)) % $(8)) * $(16384)) + (((r59_z / $(128)) % $(128)) * $(128)) + (r59_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(131072)])
    tl.store($(candidate_vmul_20_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(128))
    r60_value = (tl.cast(r60_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r60_value = tl.broadcast_to(r60_value, [$(128)])
    tl.store($(candidate_vadd_7_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(128))
    r61_value = (tl.load($(candidate_vadd_7_ptr) + r61_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r61_value = tl.broadcast_to(r61_value, [$(128)])
    tl.store($(candidate_vsub_2_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(16384))
    r62_value = (tl.cast((r62_z % $(128)), tl.int64) > tl.load($(candidate_vsub_2_ptr) + ((r62_z / $(128)) % $(128)), dtype=tl.int64))
    r62_value = tl.broadcast_to(r62_value, [$(16384)])
    tl.store($(candidate_vgt_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(131072))
    r63_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r63_z / $(128)) % $(128)) * $(128)) + (r63_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_20_ptr) + ((((r63_z / $(16384)) % $(8)) * $(16384)) + (((r63_z / $(128)) % $(128)) * $(128)) + (r63_z % $(128))), dtype=tl.float32))
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(8))
    r64_value = tl.load($(candidate_vto_copy_18_ptr) + r64_z, dtype=tl.bfloat16)
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(8)])
    tl.store($(candidate_vto_copy_27_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(132096))
    r65_value = tl.where((r65_z % $(129)) < $(128), tl.load($(candidate_vmasked_fill_2_ptr) + ((((r65_z / $(16512)) % $(8)) * $(16384)) + (((r65_z / $(129)) % $(128)) * $(128)) + (((r65_z % $(129)) - $(0)) % $(128))), dtype=tl.float32), tl.load($(candidate_vto_copy_27_ptr) + ((r65_z / $(16512)) % $(8)), dtype=tl.float32))
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(132096)])
    tl.store($(candidate_vcat_2_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(132096))
    r66_lane = tl.arange($(129))
    r66_row_values = tl.load($(candidate_vcat_2_ptr) + ((((r66_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r66_z[:, None] / $(129)) % $(128)) * $(129)) + (r66_lane[None, :] % $(129))), dtype=tl.float32)
    r66_maximum = tl.max(r66_row_values, axis=1, keep_dims=true)
    r66_weights = tl.exp(r66_row_values - r66_maximum)
    r66_total = tl.sum(r66_weights, axis=1)
    r66_maximum = tl.max(r66_row_values, axis=1)
    r66_value = tl.exp(tl.load($(candidate_vcat_2_ptr) + ((((r66_z / $(16512)) % $(8)) * $(16512)) + (((r66_z / $(129)) % $(128)) * $(129)) + (r66_z % $(129))), dtype=tl.float32) - r66_maximum) / r66_total
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(132096)])
    tl.store($(candidate_vsoftmax_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(131072))
    r67_value = tl.load($(candidate_vsoftmax_ptr) + ((((r67_z / $(16384)) % $(8)) * $(16512)) + (((r67_z / $(128)) % $(128)) * $(129)) + (r67_z % $(128))), dtype=tl.float32)
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(131072)])
    tl.store($(candidate_vto_copy_28_ptr) + r67_z, r67_value)
    r68_row = tl.arange($(1024))
    r68_col = tl.arange($(512))
    r68_lane = tl.arange($(128))
    r68_left = tl.load($(candidate_vto_copy_28_ptr) + ((((((r68_row[:, None] * $(128)) + r68_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r68_row[:, None] * $(128)) + r68_lane[None, :]) / $(128)) % $(128)) * $(128)) + (((r68_row[:, None] * $(128)) + r68_lane[None, :]) % $(128))), dtype=tl.bfloat16)
    r68_right = tl.load($(candidate_vto_copy_22_ptr) + ((((((r68_lane[:, None] * $(512)) + r68_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r68_lane[:, None] * $(512)) + r68_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r68_value = tl.dot(r68_left, r68_right)
    r68_value = r68_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r68_row[:, None] * $(512)) + r68_col[None, :]), r68_value)
    r69_z = tl.arange($(1024))
    r69_lane = tl.arange($(129))
    r69_row_values = tl.load($(candidate_vcat_2_ptr) + ((((r69_z[:, None] / $(128)) % $(8)) * $(16512)) + ((r69_z[:, None] % $(128)) * $(129)) + (r69_lane[None, :] % $(129))), dtype=tl.float32)
    r69_maximum = tl.max(r69_row_values, axis=1, keep_dims=true)
    r69_weights = tl.exp(r69_row_values - r69_maximum)
    r69_total = tl.sum(r69_weights, axis=1)
    r69_maximum = tl.max(r69_row_values, axis=1)
    r69_value = r69_maximum + tl.log(r69_total)
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(1024)])
    tl.store($(candidate_vlogsumexp_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(65536))
    r70_value = tl.load($(candidate_vcat_ptr) + ((((r70_z / $(8192)) % $(8)) * $(8192)) + (((r70_z / $(64)) % $(128)) * $(64)) + (r70_z % $(64)) + $(65536)), dtype=tl.bfloat16)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(65536)])
    tl.store($(candidate_vto_copy_29_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(8192))
    r71_value = tl.load($(candidate_vcat_1_ptr) + ((((r71_z / $(64)) % $(128)) * $(64)) + (r71_z % $(64)) + $(8192)), dtype=tl.bfloat16)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(8192)])
    tl.store($(candidate_vto_copy_30_ptr) + r71_z, r71_value)
    r72_row = tl.arange($(128))
    r72_col = tl.arange($(128))
    r72_lane = tl.arange($(64))
    for r72_group in range($(8)) {
      r72_left = tl.load($(candidate_vto_copy_29_ptr) + (((((((r72_group % $(8)) * $(8192)) + (r72_row[:, None] * $(64)) + r72_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r72_group % $(8)) * $(8192)) + (r72_row[:, None] * $(64)) + r72_lane[None, :]) / $(64)) % $(128)) * $(64)) + ((((r72_group % $(8)) * $(8192)) + (r72_row[:, None] * $(64)) + r72_lane[None, :]) % $(64))), dtype=tl.float32)
      r72_right = tl.load($(candidate_vto_copy_30_ptr) + ((((((r72_group % $(8)) * $(8192)) + (r72_lane[:, None] * $(128)) + r72_col[None, :]) % $(128)) * $(64)) + (((((r72_group % $(8)) * $(8192)) + (r72_lane[:, None] * $(128)) + r72_col[None, :]) / $(128)) % $(64))), dtype=tl.float32)
      r72_value = tl.dot(r72_left, r72_right)
      r72_value = r72_value.to(tl.float32)
      tl.store($(candidate_vbmm_2_ptr) + ((r72_group * $(16384)) + (r72_row[:, None] * $(128)) + r72_col[None, :]), r72_value)
    }
    r73_z = tl.arange($(524288))
    r73_value = tl.load($(candidate_vbmm_ptr) + ((((r73_z / $(65536)) % $(8)) * $(131072)) + (((r73_z / $(512)) % $(128)) * $(512)) + (r73_z % $(512)) + $(65536)), dtype=tl.bfloat16)
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(524288)])
    tl.store($(candidate_vto_copy_31_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(65536))
    r74_value = tl.load($(candidate_vto_copy_22_ptr) + ((((r74_z / $(512)) % $(128)) * $(512)) + (r74_z % $(512)) + $(65536)), dtype=tl.bfloat16)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(65536)])
    tl.store($(candidate_vto_copy_32_ptr) + r74_z, r74_value)
    r75_row = tl.arange($(1024))
    r75_col = tl.arange($(128))
    r75_lane = tl.arange($(512))
    r75_left = tl.load($(candidate_vto_copy_31_ptr) + ((((((r75_row[:, None] * $(512)) + r75_lane[None, :]) / $(65536)) % $(8)) * $(65536)) + (((((r75_row[:, None] * $(512)) + r75_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r75_row[:, None] * $(512)) + r75_lane[None, :]) % $(512))), dtype=tl.float32)
    r75_right = tl.load($(candidate_vto_copy_32_ptr) + (((((r75_lane[:, None] * $(128)) + r75_col[None, :]) % $(128)) * $(512)) + ((((r75_lane[:, None] * $(128)) + r75_col[None, :]) / $(128)) % $(512))), dtype=tl.float32)
    r75_value = tl.dot(r75_left, r75_right)
    r75_value = r75_value.to(tl.float32)
    tl.store($(candidate_vmm_5_ptr) + ((r75_row[:, None] * $(128)) + r75_col[None, :]), r75_value)
    r76_z = tl.arange($(131072))
    r76_value = (tl.load($(candidate_vbmm_2_ptr) + ((((r76_z / $(16384)) % $(8)) * $(16384)) + (((r76_z / $(128)) % $(128)) * $(128)) + (r76_z % $(128))), dtype=tl.float32) + tl.load($(candidate_vmm_5_ptr) + ((((r76_z / $(16384)) % $(8)) * $(16384)) + (((r76_z / $(128)) % $(128)) * $(128)) + (r76_z % $(128))), dtype=tl.float32))
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(131072)])
    tl.store($(candidate_vadd_8_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(131072))
    r77_value = (tl.load($(candidate_vadd_8_ptr) + ((((r77_z / $(16384)) % $(8)) * $(16384)) + (((r77_z / $(128)) % $(128)) * $(128)) + (r77_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(131072)])
    tl.store($(candidate_vmul_21_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(128))
    r78_value = (tl.cast(r78_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r78_value = tl.broadcast_to(r78_value, [$(128)])
    tl.store($(candidate_vadd_9_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(128))
    r79_value = (tl.load($(candidate_vadd_9_ptr) + r79_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r79_value = tl.broadcast_to(r79_value, [$(128)])
    tl.store($(candidate_vsub_3_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(16384))
    r80_value = (tl.cast((r80_z % $(128)), tl.int64) > tl.load($(candidate_vsub_3_ptr) + ((r80_z / $(128)) % $(128)), dtype=tl.int64))
    r80_value = tl.broadcast_to(r80_value, [$(16384)])
    tl.store($(candidate_vgt_1_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(131072))
    r81_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r81_z / $(128)) % $(128)) * $(128)) + (r81_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_21_ptr) + ((((r81_z / $(16384)) % $(8)) * $(16384)) + (((r81_z / $(128)) % $(128)) * $(128)) + (r81_z % $(128))), dtype=tl.float32))
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(8))
    r82_value = tl.load($(candidate_vto_copy_18_ptr) + r82_z, dtype=tl.bfloat16)
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(8)])
    tl.store($(candidate_vto_copy_33_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(132096))
    r83_value = tl.where((r83_z % $(129)) < $(128), tl.load($(candidate_vmasked_fill_3_ptr) + ((((r83_z / $(16512)) % $(8)) * $(16384)) + (((r83_z / $(129)) % $(128)) * $(128)) + (((r83_z % $(129)) - $(0)) % $(128))), dtype=tl.float32), tl.load($(candidate_vto_copy_33_ptr) + ((r83_z / $(16512)) % $(8)), dtype=tl.float32))
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(132096)])
    tl.store($(candidate_vcat_3_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(132096))
    r84_lane = tl.arange($(129))
    r84_row_values = tl.load($(candidate_vcat_3_ptr) + ((((r84_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r84_z[:, None] / $(129)) % $(128)) * $(129)) + (r84_lane[None, :] % $(129))), dtype=tl.float32)
    r84_maximum = tl.max(r84_row_values, axis=1, keep_dims=true)
    r84_weights = tl.exp(r84_row_values - r84_maximum)
    r84_total = tl.sum(r84_weights, axis=1)
    r84_maximum = tl.max(r84_row_values, axis=1)
    r84_value = tl.exp(tl.load($(candidate_vcat_3_ptr) + ((((r84_z / $(16512)) % $(8)) * $(16512)) + (((r84_z / $(129)) % $(128)) * $(129)) + (r84_z % $(129))), dtype=tl.float32) - r84_maximum) / r84_total
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(132096)])
    tl.store($(candidate_vsoftmax_1_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(131072))
    r85_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r85_z / $(16384)) % $(8)) * $(16512)) + (((r85_z / $(128)) % $(128)) * $(129)) + (r85_z % $(128))), dtype=tl.float32)
    r85_value = r85_value.to(tl.bfloat16)
    r85_value = tl.broadcast_to(r85_value, [$(131072)])
    tl.store($(candidate_vto_copy_34_ptr) + r85_z, r85_value)
    r86_row = tl.arange($(1024))
    r86_col = tl.arange($(512))
    r86_lane = tl.arange($(128))
    r86_left = tl.load($(candidate_vto_copy_34_ptr) + ((((((r86_row[:, None] * $(128)) + r86_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r86_row[:, None] * $(128)) + r86_lane[None, :]) / $(128)) % $(128)) * $(128)) + (((r86_row[:, None] * $(128)) + r86_lane[None, :]) % $(128))), dtype=tl.bfloat16)
    r86_right = tl.load($(candidate_vto_copy_22_ptr) + ((((((r86_lane[:, None] * $(512)) + r86_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r86_lane[:, None] * $(512)) + r86_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
    r86_value = tl.dot(r86_left, r86_right)
    r86_value = r86_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r86_row[:, None] * $(512)) + r86_col[None, :]), r86_value)
    r87_z = tl.arange($(1024))
    r87_lane = tl.arange($(129))
    r87_row_values = tl.load($(candidate_vcat_3_ptr) + ((((r87_z[:, None] / $(128)) % $(8)) * $(16512)) + ((r87_z[:, None] % $(128)) * $(129)) + (r87_lane[None, :] % $(129))), dtype=tl.float32)
    r87_maximum = tl.max(r87_row_values, axis=1, keep_dims=true)
    r87_weights = tl.exp(r87_row_values - r87_maximum)
    r87_total = tl.sum(r87_weights, axis=1)
    r87_maximum = tl.max(r87_row_values, axis=1)
    r87_value = r87_maximum + tl.log(r87_total)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(1024)])
    tl.store($(candidate_vlogsumexp_1_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(1048576))
    r88_value = tl.where(((r88_z / $(4096)) % $(256)) < $(128), tl.load($(candidate_vmm_4_ptr) + (((((((r88_z / $(512)) % $(8)) * $(128)) + (((r88_z / $(4096)) % $(256)) - $(0))) % $(1024)) * $(512)) + (r88_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_6_ptr) + (((((((r88_z / $(512)) % $(8)) * $(128)) + (((r88_z / $(4096)) % $(256)) - $(128))) % $(1024)) * $(512)) + (r88_z % $(512))), dtype=tl.bfloat16))
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(1048576)])
    tl.store($(candidate_vcat_4_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(2048))
    r89_value = tl.where(((r89_z / $(8)) % $(256)) < $(128), tl.load($(candidate_vlogsumexp_ptr) + (((r89_z % $(8)) * $(128)) + ((((r89_z / $(8)) % $(256)) - $(0)) % $(128))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_1_ptr) + (((r89_z % $(8)) * $(128)) + ((((r89_z / $(8)) % $(256)) - $(128)) % $(128))), dtype=tl.float32))
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(2048)])
    tl.store($(candidate_vcat_5_ptr) + r89_z, r89_value)
    r90_row = tl.arange($(256))
    r90_col = tl.arange($(128))
    r90_lane = tl.arange($(512))
    for r90_group in range($(8)) {
      r90_left = tl.load($(candidate_vcat_4_ptr) + (((((((r90_group % $(8)) * $(131072)) + (r90_row[:, None] * $(512)) + r90_lane[None, :]) / $(512)) % $(256)) * $(4096)) + ((((((r90_group % $(8)) * $(131072)) + (r90_row[:, None] * $(512)) + r90_lane[None, :]) / $(131072)) % $(8)) * $(512)) + ((((r90_group % $(8)) * $(131072)) + (r90_row[:, None] * $(512)) + r90_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r90_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r90_group % $(8)) * $(65536)) + (r90_lane[:, None] * $(128)) + r90_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r90_group % $(8)) * $(65536)) + (r90_lane[:, None] * $(128)) + r90_col[None, :]) % $(128)) * $(512)) + (((((r90_group % $(8)) * $(65536)) + (r90_lane[:, None] * $(128)) + r90_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r90_value = tl.dot(r90_left, r90_right)
      r90_value = r90_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_3_ptr) + ((r90_group * $(32768)) + (r90_row[:, None] * $(128)) + r90_col[None, :]), r90_value)
    }
    r91_row = tl.arange($(256))
    r91_col = tl.arange($(1024))
    r91_lane = tl.arange($(512))
    r91_left = tl.load($(input1_ptr) + ((((((r91_row[:, None] * $(512)) + r91_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r91_row[:, None] * $(512)) + r91_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r91_row[:, None] * $(512)) + r91_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r91_right = tl.load($(candidate_vto_copy_6_ptr) + (((r91_lane[:, None] % $(512)) * $(1024)) + (r91_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r91_value = tl.dot(r91_left, r91_right)
    r91_value = r91_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_8_ptr) + ((r91_row[:, None] * $(1024)) + r91_col[None, :]), r91_value)
    r92_z = tl.arange($(131072))
    r92_value = tl.load($(candidate_vmm_8_ptr) + ((((r92_z / $(128)) % $(128)) * $(1024)) + (((r92_z / $(16384)) % $(8)) * $(128)) + (r92_z % $(128))), dtype=tl.bfloat16)
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(131072)])
    tl.store($(candidate_vto_copy_38_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(131072))
    r93_value = tl.load($(candidate_vbmm_3_ptr) + ((((r93_z / $(16384)) % $(8)) * $(32768)) + (((r93_z / $(128)) % $(128)) * $(128)) + (r93_z % $(128))), dtype=tl.bfloat16)
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(131072)])
    tl.store($(candidate_vto_copy_39_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(131072))
    r94_value = (tl.load($(candidate_vto_copy_38_ptr) + ((((r94_z / $(16384)) % $(8)) * $(16384)) + (((r94_z / $(128)) % $(128)) * $(128)) + (r94_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_39_ptr) + ((((r94_z / $(16384)) % $(8)) * $(16384)) + (((r94_z / $(128)) % $(128)) * $(128)) + (r94_z % $(128))), dtype=tl.float32))
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(131072)])
    tl.store($(candidate_vmul_23_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(1024))
    r95_lane = tl.arange($(128))
    r95_value = tl.sum(tl.load($(candidate_vmul_23_ptr) + ((((r95_z[:, None] / $(128)) % $(8)) * $(16384)) + ((r95_z[:, None] % $(128)) * $(128)) + r95_lane[None, :]), dtype=tl.float32), axis=1)
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(1024)])
    tl.store($(candidate_vsum_1_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(8))
    r96_value = tl.load($(candidate_vto_copy_18_ptr) + r96_z, dtype=tl.bfloat16)
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(8)])
    tl.store($(candidate_vto_copy_43_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(1024))
    r97_value = (tl.load($(candidate_vto_copy_43_ptr) + ((r97_z / $(128)) % $(8)), dtype=tl.float32) - tl.load($(candidate_vcat_5_ptr) + (((r97_z % $(128)) * $(8)) + ((r97_z / $(128)) % $(8))), dtype=tl.float32))
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(1024)])
    tl.store($(candidate_vsub_7_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(1024))
    r98_value = tl.exp(tl.load($(candidate_vsub_7_ptr) + ((((r98_z / $(128)) % $(8)) * $(128)) + (r98_z % $(128))), dtype=tl.float32))
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(1024)])
    tl.store($(candidate_vexp_1_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(1024))
    r99_value = (tl.load($(candidate_vexp_1_ptr) + ((((r99_z / $(128)) % $(8)) * $(128)) + (r99_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vsum_1_ptr) + ((((r99_z / $(128)) % $(8)) * $(128)) + (r99_z % $(128))), dtype=tl.float32))
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(1024)])
    tl.store($(candidate_vmul_27_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(8))
    r100_lane = tl.arange($(128))
    r100_value = tl.sum(tl.load($(candidate_vmul_27_ptr) + ((r100_z[:, None] * $(128)) + r100_lane[None, :]), dtype=tl.float32), axis=1)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(8)])
    tl.store($(candidate_vsum_2_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(8))
    r101_value = -tl.load($(candidate_vsum_2_ptr) + r101_z, dtype=tl.float32)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(8)])
    tl.store($(candidate_vneg_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(131072))
    r102_value = tl.load($(candidate_vmm_8_ptr) + ((((r102_z / $(128)) % $(128)) * $(1024)) + (((r102_z / $(16384)) % $(8)) * $(128)) + (r102_z % $(128)) + $(131072)), dtype=tl.bfloat16)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(131072)])
    tl.store($(candidate_vto_copy_47_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(131072))
    r103_value = tl.load($(candidate_vbmm_3_ptr) + ((((r103_z / $(16384)) % $(8)) * $(32768)) + (((r103_z / $(128)) % $(128)) * $(128)) + (r103_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(131072)])
    tl.store($(candidate_vto_copy_48_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(131072))
    r104_value = (tl.load($(candidate_vto_copy_47_ptr) + ((((r104_z / $(16384)) % $(8)) * $(16384)) + (((r104_z / $(128)) % $(128)) * $(128)) + (r104_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_48_ptr) + ((((r104_z / $(16384)) % $(8)) * $(16384)) + (((r104_z / $(128)) % $(128)) * $(128)) + (r104_z % $(128))), dtype=tl.float32))
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(131072)])
    tl.store($(candidate_vmul_29_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(1024))
    r105_lane = tl.arange($(128))
    r105_value = tl.sum(tl.load($(candidate_vmul_29_ptr) + ((((r105_z[:, None] / $(128)) % $(8)) * $(16384)) + ((r105_z[:, None] % $(128)) * $(128)) + r105_lane[None, :]), dtype=tl.float32), axis=1)
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(1024)])
    tl.store($(candidate_vsum_3_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(8))
    r106_value = tl.load($(candidate_vto_copy_18_ptr) + r106_z, dtype=tl.bfloat16)
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(8)])
    tl.store($(candidate_vto_copy_52_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(1024))
    r107_value = (tl.load($(candidate_vto_copy_52_ptr) + ((r107_z / $(128)) % $(8)), dtype=tl.float32) - tl.load($(candidate_vcat_5_ptr) + (((r107_z % $(128)) * $(8)) + ((r107_z / $(128)) % $(8)) + $(1024)), dtype=tl.float32))
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(1024)])
    tl.store($(candidate_vsub_11_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(1024))
    r108_value = tl.exp(tl.load($(candidate_vsub_11_ptr) + ((((r108_z / $(128)) % $(8)) * $(128)) + (r108_z % $(128))), dtype=tl.float32))
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(1024)])
    tl.store($(candidate_vexp_3_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(1024))
    r109_value = (tl.load($(candidate_vexp_3_ptr) + ((((r109_z / $(128)) % $(8)) * $(128)) + (r109_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vsum_3_ptr) + ((((r109_z / $(128)) % $(8)) * $(128)) + (r109_z % $(128))), dtype=tl.float32))
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(1024)])
    tl.store($(candidate_vmul_33_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(8))
    r110_lane = tl.arange($(128))
    r110_value = tl.sum(tl.load($(candidate_vmul_33_ptr) + ((r110_z[:, None] * $(128)) + r110_lane[None, :]), dtype=tl.float32), axis=1)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(8)])
    tl.store($(candidate_vsum_4_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(8))
    r111_value = -tl.load($(candidate_vsum_4_ptr) + r111_z, dtype=tl.float32)
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(8)])
    tl.store($(candidate_vneg_1_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(8))
    r112_value = (tl.load($(candidate_vneg_ptr) + r112_z, dtype=tl.float32) + 0.0)
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(8)])
    tl.store($(candidate_vadd_12_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(8))
    r113_value = (tl.load($(candidate_vadd_12_ptr) + r113_z, dtype=tl.float32) + tl.load($(candidate_vneg_1_ptr) + r113_z, dtype=tl.float32))
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(8)])
    tl.store($(candidate_vadd_13_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(8))
    r114_value = tl.load($(candidate_vadd_13_ptr) + r114_z, dtype=tl.float32)
    r114_value = r114_value.to(tl.bfloat16)
    r114_value = tl.broadcast_to(r114_value, [$(8)])
    tl.store($(candidate_vto_copy_53_ptr) + r114_z, r114_value)
    z = tl.arange($(8))
    value = tl.load($(candidate_vto_copy_53_ptr) + z, dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(8)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_reuse_sink_dsink
