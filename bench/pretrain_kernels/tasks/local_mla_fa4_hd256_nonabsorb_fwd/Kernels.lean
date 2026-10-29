import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_nonabsorb_fwd
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
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vsub_ptr : RegionName := "vsub"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_ptr : RegionName := "vcat"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vbmm_1_ptr : RegionName := "vbmm_1"
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
    r14_z = tl.arange($(917504))
    r14_value = tl.load($(vmul_5_ptr) + ((((r14_z / $(256)) % $(3584)) * $(256)) + (r14_z % $(256))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(917504)])
    tl.store($(vto_copy_5_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(1048576))
    r15_value = tl.load($(vmul_6_ptr) + ((((r15_z / $(2048)) % $(512)) * $(2048)) + (r15_z % $(2048))), dtype=tl.float32)
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(1048576)])
    tl.store($(vto_copy_6_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(16384))
    r16_value = tl.where((r16_z % $(128)) >= ((r16_z / $(128)) % $(128)) + $(1), true, false)
    r16_value = tl.broadcast_to(r16_value, [$(16384)])
    tl.store($(vtriu_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_value = tl.where(tl.load($(vtriu_ptr) + ((((r17_z / $(128)) % $(128)) * $(128)) + (r17_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(vmasked_fill_ptr) + r17_z, r17_value)
    r18_row = tl.arange($(256))
    r18_col = tl.arange($(256))
    r18_lane = tl.arange($(512))
    r18_left = tl.load($(input0_ptr) + ((((((r18_row[:, None] * $(512)) + r18_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r18_row[:, None] * $(512)) + r18_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r18_row[:, None] * $(512)) + r18_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r18_right = tl.load($(vto_copy_ptr) + (((r18_col[None, :] % $(256)) * $(512)) + (r18_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r18_value = tl.dot(r18_left, r18_right)
    r18_value = r18_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r18_row[:, None] * $(256)) + r18_col[None, :]), r18_value)
    r19_z = tl.arange($(65536))
    r19_value = tl.load($(vmm_ptr) + ((((r19_z / $(32768)) % $(2)) * $(32768)) + (((r19_z / $(256)) % $(128)) * $(256)) + (r19_z % $(256))), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(65536)])
    tl.store($(vto_copy_16_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(65536))
    r20_value = ((tl.load($(vto_copy_16_ptr) + ((((r20_z / $(32768)) % $(2)) * $(32768)) + (((r20_z / $(256)) % $(128)) * $(256)) + (r20_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_16_ptr) + ((((r20_z / $(32768)) % $(2)) * $(32768)) + (((r20_z / $(256)) % $(128)) * $(256)) + (r20_z % $(256))), dtype=tl.float32)))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(65536)])
    tl.store($(vpow_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(256))
    r21_lane = tl.arange($(256))
    r21_value = (tl.sum(tl.load($(vpow_1_ptr) + ((((r21_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r21_z[:, None] % $(128)) * $(256)) + r21_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(256)])
    tl.store($(vmean_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(256))
    r22_value = (tl.load($(vmean_ptr) + ((((r22_z / $(128)) % $(2)) * $(128)) + (r22_z % $(128))), dtype=tl.float32) + 0.000001)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(256)])
    tl.store($(vadd_2_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(256))
    r23_value = tl.rsqrt(tl.load($(vadd_2_ptr) + ((((r23_z / $(128)) % $(2)) * $(128)) + (r23_z % $(128))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(256)])
    tl.store($(vrsqrt_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = (tl.load($(vto_copy_16_ptr) + ((((r24_z / $(32768)) % $(2)) * $(32768)) + (((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r24_z / $(32768)) % $(2)) * $(128)) + ((r24_z / $(256)) % $(128))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(vmul_7_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(65536))
    r25_value = (tl.cast(tl.load($(vto_copy_1_ptr) + (r25_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_7_ptr) + ((((r25_z / $(32768)) % $(2)) * $(32768)) + (((r25_z / $(256)) % $(128)) * $(256)) + (r25_z % $(256))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(65536)])
    tl.store($(vmul_8_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(65536))
    r26_value = tl.load($(vmul_8_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(256)) % $(128)) * $(256)) + (r26_z % $(256))), dtype=tl.float32)
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(65536)])
    tl.store($(vto_copy_17_ptr) + r26_z, r26_value)
    r27_row = tl.arange($(256))
    r27_col = tl.arange($(2048))
    r27_lane = tl.arange($(256))
    r27_left = tl.load($(vto_copy_17_ptr) + ((((((r27_row[:, None] * $(256)) + r27_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r27_row[:, None] * $(256)) + r27_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r27_row[:, None] * $(256)) + r27_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r27_right = tl.load($(vto_copy_2_ptr) + (((r27_col[None, :] % $(2048)) * $(256)) + (r27_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r27_value = tl.dot(r27_left, r27_right)
    r27_value = r27_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r27_row[:, None] * $(2048)) + r27_col[None, :]), r27_value)
    r28_row = tl.arange($(256))
    r28_col = tl.arange($(320))
    r28_lane = tl.arange($(512))
    r28_left = tl.load($(input0_ptr) + ((((((r28_row[:, None] * $(512)) + r28_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r28_row[:, None] * $(512)) + r28_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r28_row[:, None] * $(512)) + r28_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r28_right = tl.load($(vto_copy_3_ptr) + (((r28_col[None, :] % $(320)) * $(512)) + (r28_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r28_value = tl.dot(r28_left, r28_right)
    r28_value = r28_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r28_row[:, None] * $(320)) + r28_col[None, :]), r28_value)
    r29_z = tl.arange($(65536))
    r29_value = tl.load($(vmm_2_ptr) + ((((r29_z / $(32768)) % $(2)) * $(40960)) + (((r29_z / $(256)) % $(128)) * $(320)) + (r29_z % $(256))), dtype=tl.bfloat16)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(65536)])
    tl.store($(vto_copy_18_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(65536))
    r30_value = ((tl.load($(vto_copy_18_ptr) + ((((r30_z / $(32768)) % $(2)) * $(32768)) + (((r30_z / $(256)) % $(128)) * $(256)) + (r30_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_18_ptr) + ((((r30_z / $(32768)) % $(2)) * $(32768)) + (((r30_z / $(256)) % $(128)) * $(256)) + (r30_z % $(256))), dtype=tl.float32)))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(65536)])
    tl.store($(vpow_2_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(256))
    r31_lane = tl.arange($(256))
    r31_value = (tl.sum(tl.load($(vpow_2_ptr) + ((((r31_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r31_z[:, None] % $(128)) * $(256)) + r31_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(256)])
    tl.store($(vmean_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(256))
    r32_value = (tl.load($(vmean_1_ptr) + ((((r32_z / $(128)) % $(2)) * $(128)) + (r32_z % $(128))), dtype=tl.float32) + 0.000001)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(256)])
    tl.store($(vadd_3_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(256))
    r33_value = tl.rsqrt(tl.load($(vadd_3_ptr) + ((((r33_z / $(128)) % $(2)) * $(128)) + (r33_z % $(128))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(256)])
    tl.store($(vrsqrt_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(65536))
    r34_value = (tl.load($(vto_copy_18_ptr) + ((((r34_z / $(32768)) % $(2)) * $(32768)) + (((r34_z / $(256)) % $(128)) * $(256)) + (r34_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((((r34_z / $(32768)) % $(2)) * $(128)) + ((r34_z / $(256)) % $(128))), dtype=tl.float32))
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(65536)])
    tl.store($(vmul_9_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(65536))
    r35_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r35_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_9_ptr) + ((((r35_z / $(32768)) % $(2)) * $(32768)) + (((r35_z / $(256)) % $(128)) * $(256)) + (r35_z % $(256))), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(65536)])
    tl.store($(vmul_10_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(65536))
    r36_value = tl.load($(vmul_10_ptr) + ((((r36_z / $(32768)) % $(2)) * $(32768)) + (((r36_z / $(256)) % $(128)) * $(256)) + (r36_z % $(256))), dtype=tl.float32)
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(65536)])
    tl.store($(vto_copy_19_ptr) + r36_z, r36_value)
    r37_row = tl.arange($(256))
    r37_col = tl.arange($(3584))
    r37_lane = tl.arange($(256))
    r37_left = tl.load($(vto_copy_19_ptr) + ((((((r37_row[:, None] * $(256)) + r37_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r37_row[:, None] * $(256)) + r37_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r37_row[:, None] * $(256)) + r37_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r37_right = tl.load($(vto_copy_5_ptr) + (((r37_col[None, :] % $(3584)) * $(256)) + (r37_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r37_value = tl.dot(r37_left, r37_right)
    r37_value = r37_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r37_row[:, None] * $(3584)) + r37_col[None, :]), r37_value)
    r38_z = tl.arange($(65536))
    r38_value = (tl.load($(vmm_1_ptr) + ((((r38_z / $(32768)) % $(2)) * $(262144)) + (((r38_z / $(32)) % $(128)) * $(2048)) + (((r38_z / $(4096)) % $(8)) * $(256)) + ((r38_z % $(32)) * $(2)) + $(192)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(65536)])
    tl.store($(vmul_11_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(65536))
    r39_value = (tl.load($(vmm_1_ptr) + ((((r39_z / $(32768)) % $(2)) * $(262144)) + (((r39_z / $(32)) % $(128)) * $(2048)) + (((r39_z / $(4096)) % $(8)) * $(256)) + ((r39_z % $(32)) * $(2)) + $(193)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(65536)])
    tl.store($(vmul_12_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(65536))
    r40_value = (tl.load($(vmul_11_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(4096)) % $(8)) * $(4096)) + (((r40_z / $(32)) % $(128)) * $(32)) + (r40_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_12_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(4096)) % $(8)) * $(4096)) + (((r40_z / $(32)) % $(128)) * $(32)) + (r40_z % $(32))), dtype=tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(vsub_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(65536))
    r41_value = (tl.load($(vmm_1_ptr) + ((((r41_z / $(32768)) % $(2)) * $(262144)) + (((r41_z / $(32)) % $(128)) * $(2048)) + (((r41_z / $(4096)) % $(8)) * $(256)) + ((r41_z % $(32)) * $(2)) + $(193)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(65536)])
    tl.store($(vmul_13_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(65536))
    r42_value = (tl.load($(vmm_1_ptr) + ((((r42_z / $(32768)) % $(2)) * $(262144)) + (((r42_z / $(32)) % $(128)) * $(2048)) + (((r42_z / $(4096)) % $(8)) * $(256)) + ((r42_z % $(32)) * $(2)) + $(192)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(65536)])
    tl.store($(vmul_14_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(65536))
    r43_value = (tl.load($(vmul_13_ptr) + ((((r43_z / $(32768)) % $(2)) * $(32768)) + (((r43_z / $(4096)) % $(8)) * $(4096)) + (((r43_z / $(32)) % $(128)) * $(32)) + (r43_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_14_ptr) + ((((r43_z / $(32768)) % $(2)) * $(32768)) + (((r43_z / $(4096)) % $(8)) * $(4096)) + (((r43_z / $(32)) % $(128)) * $(32)) + (r43_z % $(32))), dtype=tl.bfloat16))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(65536)])
    tl.store($(vadd_4_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(131072))
    r44_value = tl.where((r44_z % $(64)) < $(32), tl.load($(vsub_ptr) + ((((r44_z / $(65536)) % $(2)) * $(32768)) + (((r44_z / $(8192)) % $(8)) * $(4096)) + (((r44_z / $(64)) % $(128)) * $(32)) + (((r44_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_4_ptr) + ((((r44_z / $(65536)) % $(2)) * $(32768)) + (((r44_z / $(8192)) % $(8)) * $(4096)) + (((r44_z / $(64)) % $(128)) * $(32)) + (((r44_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(131072)])
    tl.store($(vcat_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(8192))
    r45_value = (tl.load($(vmm_2_ptr) + ((((r45_z / $(4096)) % $(2)) * $(40960)) + (((r45_z / $(32)) % $(128)) * $(320)) + ((r45_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(8192)])
    tl.store($(vmul_15_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(8192))
    r46_value = (tl.load($(vmm_2_ptr) + ((((r46_z / $(4096)) % $(2)) * $(40960)) + (((r46_z / $(32)) % $(128)) * $(320)) + ((r46_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(8192)])
    tl.store($(vmul_16_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(8192))
    r47_value = (tl.load($(vmul_15_ptr) + ((((r47_z / $(4096)) % $(2)) * $(4096)) + (((r47_z / $(32)) % $(128)) * $(32)) + (r47_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_16_ptr) + ((((r47_z / $(4096)) % $(2)) * $(4096)) + (((r47_z / $(32)) % $(128)) * $(32)) + (r47_z % $(32))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(8192)])
    tl.store($(vsub_1_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(8192))
    r48_value = (tl.load($(vmm_2_ptr) + ((((r48_z / $(4096)) % $(2)) * $(40960)) + (((r48_z / $(32)) % $(128)) * $(320)) + ((r48_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(8192)])
    tl.store($(vmul_17_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(8192))
    r49_value = (tl.load($(vmm_2_ptr) + ((((r49_z / $(4096)) % $(2)) * $(40960)) + (((r49_z / $(32)) % $(128)) * $(320)) + ((r49_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(8192)])
    tl.store($(vmul_18_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(8192))
    r50_value = (tl.load($(vmul_17_ptr) + ((((r50_z / $(4096)) % $(2)) * $(4096)) + (((r50_z / $(32)) % $(128)) * $(32)) + (r50_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_18_ptr) + ((((r50_z / $(4096)) % $(2)) * $(4096)) + (((r50_z / $(32)) % $(128)) * $(32)) + (r50_z % $(32))), dtype=tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(8192)])
    tl.store($(vadd_5_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(16384))
    r51_value = tl.where((r51_z % $(64)) < $(32), tl.load($(vsub_1_ptr) + ((((r51_z / $(8192)) % $(2)) * $(4096)) + (((r51_z / $(64)) % $(128)) * $(32)) + (((r51_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r51_z / $(8192)) % $(2)) * $(4096)) + (((r51_z / $(64)) % $(128)) * $(32)) + (((r51_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(16384)])
    tl.store($(vcat_1_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(524288))
    r52_value = tl.where((r52_z % $(256)) < $(192), tl.load($(vmm_1_ptr) + ((((((((r52_z / $(262144)) % $(2)) * $(262144)) + (((r52_z / $(256)) % $(128)) * $(2048)) + (((r52_z / $(32768)) % $(8)) * $(256)) + ((r52_z % $(256)) - $(0))) / $(262144)) % $(2)) * $(262144)) + (((((((r52_z / $(262144)) % $(2)) * $(262144)) + (((r52_z / $(256)) % $(128)) * $(2048)) + (((r52_z / $(32768)) % $(8)) * $(256)) + ((r52_z % $(256)) - $(0))) / $(2048)) % $(128)) * $(2048)) + (((((r52_z / $(262144)) % $(2)) * $(262144)) + (((r52_z / $(256)) % $(128)) * $(2048)) + (((r52_z / $(32768)) % $(8)) * $(256)) + ((r52_z % $(256)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_ptr) + ((((r52_z / $(262144)) % $(2)) * $(65536)) + (((r52_z / $(32768)) % $(8)) * $(8192)) + (((r52_z / $(256)) % $(128)) * $(64)) + (((r52_z % $(256)) - $(192)) % $(64))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(524288)])
    tl.store($(vcat_2_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(524288))
    r53_value = tl.where((r53_z % $(256)) < $(192), tl.load($(vmm_3_ptr) + ((((((((r53_z / $(262144)) % $(2)) * $(458752)) + (((r53_z / $(256)) % $(128)) * $(3584)) + (((r53_z / $(32768)) % $(8)) * $(448)) + ((r53_z % $(256)) - $(0))) / $(458752)) % $(2)) * $(458752)) + (((((((r53_z / $(262144)) % $(2)) * $(458752)) + (((r53_z / $(256)) % $(128)) * $(3584)) + (((r53_z / $(32768)) % $(8)) * $(448)) + ((r53_z % $(256)) - $(0))) / $(3584)) % $(128)) * $(3584)) + (((((r53_z / $(262144)) % $(2)) * $(458752)) + (((r53_z / $(256)) % $(128)) * $(3584)) + (((r53_z / $(32768)) % $(8)) * $(448)) + ((r53_z % $(256)) - $(0))) % $(3584))), dtype=tl.bfloat16), tl.load($(vcat_1_ptr) + ((((r53_z / $(262144)) % $(2)) * $(8192)) + (((r53_z / $(256)) % $(128)) * $(64)) + (((r53_z % $(256)) - $(192)) % $(64))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(524288)])
    tl.store($(vcat_3_ptr) + r53_z, r53_value)
    r54_row = tl.arange($(128))
    r54_col = tl.arange($(128))
    r54_lane = tl.arange($(256))
    for r54_group in range($(16)) {
      r54_left = tl.load($(vcat_2_ptr) + (((((((r54_group % $(16)) * $(32768)) + (r54_row[:, None] * $(256)) + r54_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + ((((((r54_group % $(16)) * $(32768)) + (r54_row[:, None] * $(256)) + r54_lane[None, :]) / $(32768)) % $(8)) * $(32768)) + ((((((r54_group % $(16)) * $(32768)) + (r54_row[:, None] * $(256)) + r54_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r54_group % $(16)) * $(32768)) + (r54_row[:, None] * $(256)) + r54_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r54_right = tl.load($(vcat_3_ptr) + (((((((r54_group % $(16)) * $(32768)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) / $(262144)) % $(2)) * $(262144)) + ((((((r54_group % $(16)) * $(32768)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) / $(32768)) % $(8)) * $(32768)) + (((((r54_group % $(16)) * $(32768)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) % $(128)) * $(256)) + (((((r54_group % $(16)) * $(32768)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) / $(128)) % $(256))), dtype=tl.bfloat16)
      r54_value = tl.dot(r54_left, r54_right)
      r54_value = r54_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r54_group * $(16384)) + (r54_row[:, None] * $(128)) + r54_col[None, :]), r54_value)
    }
    r55_z = tl.arange($(262144))
    r55_value = (tl.load($(vbmm_ptr) + ((((r55_z / $(131072)) % $(2)) * $(131072)) + (((r55_z / $(16384)) % $(8)) * $(16384)) + (((r55_z / $(128)) % $(128)) * $(128)) + (r55_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.0625, tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(262144)])
    tl.store($(vmul_19_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(262144))
    r56_value = (tl.load($(vmul_19_ptr) + ((((r56_z / $(131072)) % $(2)) * $(131072)) + (((r56_z / $(16384)) % $(8)) * $(16384)) + (((r56_z / $(128)) % $(128)) * $(128)) + (r56_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r56_z / $(131072)) % $(2)) * $(16384)) + (((r56_z / $(128)) % $(128)) * $(128)) + (r56_z % $(128))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(262144)])
    tl.store($(vadd_6_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(262144))
    r57_value = tl.load($(vadd_6_ptr) + ((((r57_z / $(131072)) % $(2)) * $(131072)) + (((r57_z / $(16384)) % $(8)) * $(16384)) + (((r57_z / $(128)) % $(128)) * $(128)) + (r57_z % $(128))), dtype=tl.bfloat16)
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(262144)])
    tl.store($(vto_copy_20_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(262144))
    r58_lane = tl.arange($(128))
    r58_row_values = tl.load($(vto_copy_20_ptr) + ((((r58_z[:, None] / $(131072)) % $(2)) * $(131072)) + (((r58_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r58_z[:, None] / $(128)) % $(128)) * $(128)) + (r58_lane[None, :] % $(128))), dtype=tl.float32)
    r58_maximum = tl.max(r58_row_values, axis=1, keep_dims=true)
    r58_weights = tl.exp(r58_row_values - r58_maximum)
    r58_total = tl.sum(r58_weights, axis=1)
    r58_maximum = tl.max(r58_row_values, axis=1)
    r58_value = tl.exp(tl.load($(vto_copy_20_ptr) + ((((r58_z / $(131072)) % $(2)) * $(131072)) + (((r58_z / $(16384)) % $(8)) * $(16384)) + (((r58_z / $(128)) % $(128)) * $(128)) + (r58_z % $(128))), dtype=tl.float32) - r58_maximum) / r58_total
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(262144)])
    tl.store($(vsoftmax_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(262144))
    r59_value = tl.load($(vsoftmax_ptr) + ((((r59_z / $(131072)) % $(2)) * $(131072)) + (((r59_z / $(16384)) % $(8)) * $(16384)) + (((r59_z / $(128)) % $(128)) * $(128)) + (r59_z % $(128))), dtype=tl.float32)
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(262144)])
    tl.store($(vto_copy_21_ptr) + r59_z, r59_value)
    r60_row = tl.arange($(128))
    r60_col = tl.arange($(256))
    r60_lane = tl.arange($(128))
    for r60_group in range($(16)) {
      r60_left = tl.load($(vto_copy_21_ptr) + (((((((r60_group % $(16)) * $(16384)) + (r60_row[:, None] * $(128)) + r60_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r60_group % $(16)) * $(16384)) + (r60_row[:, None] * $(128)) + r60_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r60_group % $(16)) * $(16384)) + (r60_row[:, None] * $(128)) + r60_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r60_group % $(16)) * $(16384)) + (r60_row[:, None] * $(128)) + r60_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r60_right = tl.load($(vmm_3_ptr) + (((((((r60_group % $(16)) * $(32768)) + (r60_lane[:, None] * $(256)) + r60_col[None, :]) / $(262144)) % $(2)) * $(458752)) + ((((((r60_group % $(16)) * $(32768)) + (r60_lane[:, None] * $(256)) + r60_col[None, :]) / $(256)) % $(128)) * $(3584)) + ((((((r60_group % $(16)) * $(32768)) + (r60_lane[:, None] * $(256)) + r60_col[None, :]) / $(32768)) % $(8)) * $(448)) + ((((r60_group % $(16)) * $(32768)) + (r60_lane[:, None] * $(256)) + r60_col[None, :]) % $(256)) + $(192)), dtype=tl.bfloat16)
      r60_value = tl.dot(r60_left, r60_right)
      r60_value = r60_value.to(tl.bfloat16)
      tl.store($(vbmm_1_ptr) + ((r60_group * $(32768)) + (r60_row[:, None] * $(256)) + r60_col[None, :]), r60_value)
    }
    r61_row = tl.arange($(256))
    r61_col = tl.arange($(512))
    r61_lane = tl.arange($(2048))
    r61_left = tl.load($(vbmm_1_ptr) + ((((((r61_row[:, None] * $(2048)) + r61_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((((((r61_row[:, None] * $(2048)) + r61_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r61_row[:, None] * $(2048)) + r61_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r61_row[:, None] * $(2048)) + r61_lane[None, :]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r61_row[:, None] * $(2048)) + r61_lane[None, :]) / $(2048)) % $(128)) * $(256)) + (((((((r61_row[:, None] * $(2048)) + r61_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r61_row[:, None] * $(2048)) + r61_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r61_row[:, None] * $(2048)) + r61_lane[None, :]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r61_right = tl.load($(vto_copy_6_ptr) + (((r61_col[None, :] % $(512)) * $(2048)) + (r61_lane[:, None] % $(2048))), dtype=tl.bfloat16)
    r61_value = tl.dot(r61_left, r61_right)
    r61_value = r61_value.to(tl.bfloat16)
    tl.store($(vmm_4_ptr) + ((r61_row[:, None] * $(512)) + r61_col[None, :]), r61_value)
    z = tl.arange($(131072))
    value = tl.load($(vmm_4_ptr) + ((((z / $(65536)) % $(2)) * $(65536)) + (((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
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
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vsub_ptr : RegionName := "vsub"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_ptr : RegionName := "vcat"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vbmm_1_ptr : RegionName := "vbmm_1"
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
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_16_ptr : RegionName := "candidate_vto_copy_16"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vto_copy_17_ptr : RegionName := "candidate_vto_copy_17"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vpow_2_ptr : RegionName := "candidate_vpow_2"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vcat_1_ptr : RegionName := "candidate_vcat_1"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vcat_2_ptr : RegionName := "candidate_vcat_2"
  let candidate_vcat_3_ptr : RegionName := "candidate_vcat_3"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vsub_2_ptr : RegionName := "candidate_vsub_2"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_3_ptr : RegionName := "candidate_vsub_3"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vcat_4_ptr : RegionName := "candidate_vcat_4"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
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
    r14_z = tl.arange($(917504))
    r14_value = tl.load($(candidate_vmul_5_ptr) + ((((r14_z / $(256)) % $(3584)) * $(256)) + (r14_z % $(256))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(917504)])
    tl.store($(candidate_vto_copy_5_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(1048576))
    r15_value = tl.load($(candidate_vmul_6_ptr) + ((((r15_z / $(2048)) % $(512)) * $(2048)) + (r15_z % $(2048))), dtype=tl.float32)
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(1048576)])
    tl.store($(candidate_vto_copy_6_ptr) + r15_z, r15_value)
    r16_row = tl.arange($(256))
    r16_col = tl.arange($(256))
    r16_lane = tl.arange($(512))
    r16_left = tl.load($(input0_ptr) + ((((((r16_row[:, None] * $(512)) + r16_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r16_row[:, None] * $(512)) + r16_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r16_row[:, None] * $(512)) + r16_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r16_right = tl.load($(candidate_vto_copy_ptr) + (((r16_col[None, :] % $(256)) * $(512)) + (r16_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r16_value = tl.dot(r16_left, r16_right)
    r16_value = r16_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r16_row[:, None] * $(256)) + r16_col[None, :]), r16_value)
    r17_z = tl.arange($(65536))
    r17_value = tl.load($(candidate_vmm_ptr) + ((((r17_z / $(32768)) % $(2)) * $(32768)) + (((r17_z / $(256)) % $(128)) * $(256)) + (r17_z % $(256))), dtype=tl.bfloat16)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(65536)])
    tl.store($(candidate_vto_copy_16_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(65536))
    r18_value = ((tl.load($(candidate_vto_copy_16_ptr) + ((((r18_z / $(32768)) % $(2)) * $(32768)) + (((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_16_ptr) + ((((r18_z / $(32768)) % $(2)) * $(32768)) + (((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(65536)])
    tl.store($(candidate_vpow_1_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(256))
    r19_lane = tl.arange($(256))
    r19_value = (tl.sum(tl.load($(candidate_vpow_1_ptr) + ((((r19_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r19_z[:, None] % $(128)) * $(256)) + r19_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(256)])
    tl.store($(candidate_vmean_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(256))
    r20_value = (tl.load($(candidate_vmean_ptr) + ((((r20_z / $(128)) % $(2)) * $(128)) + (r20_z % $(128))), dtype=tl.float32) + 0.000001)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(256)])
    tl.store($(candidate_vadd_2_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(256))
    r21_value = tl.rsqrt(tl.load($(candidate_vadd_2_ptr) + ((((r21_z / $(128)) % $(2)) * $(128)) + (r21_z % $(128))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(256)])
    tl.store($(candidate_vrsqrt_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(65536))
    r22_value = (tl.load($(candidate_vto_copy_16_ptr) + ((((r22_z / $(32768)) % $(2)) * $(32768)) + (((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r22_z / $(32768)) % $(2)) * $(128)) + ((r22_z / $(256)) % $(128))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(65536)])
    tl.store($(candidate_vmul_7_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(65536))
    r23_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r23_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_7_ptr) + ((((r23_z / $(32768)) % $(2)) * $(32768)) + (((r23_z / $(256)) % $(128)) * $(256)) + (r23_z % $(256))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(65536)])
    tl.store($(candidate_vmul_8_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = tl.load($(candidate_vmul_8_ptr) + ((((r24_z / $(32768)) % $(2)) * $(32768)) + (((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.float32)
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(candidate_vto_copy_17_ptr) + r24_z, r24_value)
    r25_row = tl.arange($(256))
    r25_col = tl.arange($(2048))
    r25_lane = tl.arange($(256))
    r25_left = tl.load($(candidate_vto_copy_17_ptr) + ((((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r25_row[:, None] * $(256)) + r25_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r25_right = tl.load($(candidate_vto_copy_2_ptr) + (((r25_col[None, :] % $(2048)) * $(256)) + (r25_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r25_value = tl.dot(r25_left, r25_right)
    r25_value = r25_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r25_row[:, None] * $(2048)) + r25_col[None, :]), r25_value)
    r26_row = tl.arange($(256))
    r26_col = tl.arange($(320))
    r26_lane = tl.arange($(512))
    r26_left = tl.load($(input0_ptr) + ((((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r26_row[:, None] * $(512)) + r26_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r26_right = tl.load($(candidate_vto_copy_3_ptr) + (((r26_col[None, :] % $(320)) * $(512)) + (r26_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r26_value = tl.dot(r26_left, r26_right)
    r26_value = r26_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r26_row[:, None] * $(320)) + r26_col[None, :]), r26_value)
    r27_z = tl.arange($(65536))
    r27_value = tl.load($(candidate_vmm_2_ptr) + ((((r27_z / $(32768)) % $(2)) * $(40960)) + (((r27_z / $(256)) % $(128)) * $(320)) + (r27_z % $(256))), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(65536)])
    tl.store($(candidate_vto_copy_18_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(65536))
    r28_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r28_z / $(32768)) % $(2)) * $(32768)) + (((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_18_ptr) + ((((r28_z / $(32768)) % $(2)) * $(32768)) + (((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(65536)])
    tl.store($(candidate_vpow_2_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(256))
    r29_lane = tl.arange($(256))
    r29_value = (tl.sum(tl.load($(candidate_vpow_2_ptr) + ((((r29_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r29_z[:, None] % $(128)) * $(256)) + r29_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(256)])
    tl.store($(candidate_vmean_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(256))
    r30_value = (tl.load($(candidate_vmean_1_ptr) + ((((r30_z / $(128)) % $(2)) * $(128)) + (r30_z % $(128))), dtype=tl.float32) + 0.000001)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(256)])
    tl.store($(candidate_vadd_3_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(256))
    r31_value = tl.rsqrt(tl.load($(candidate_vadd_3_ptr) + ((((r31_z / $(128)) % $(2)) * $(128)) + (r31_z % $(128))), dtype=tl.float32))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(256)])
    tl.store($(candidate_vrsqrt_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(65536))
    r32_value = (tl.load($(candidate_vto_copy_18_ptr) + ((((r32_z / $(32768)) % $(2)) * $(32768)) + (((r32_z / $(256)) % $(128)) * $(256)) + (r32_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((((r32_z / $(32768)) % $(2)) * $(128)) + ((r32_z / $(256)) % $(128))), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(65536)])
    tl.store($(candidate_vmul_9_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(65536))
    r33_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r33_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_9_ptr) + ((((r33_z / $(32768)) % $(2)) * $(32768)) + (((r33_z / $(256)) % $(128)) * $(256)) + (r33_z % $(256))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(65536)])
    tl.store($(candidate_vmul_10_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(65536))
    r34_value = tl.load($(candidate_vmul_10_ptr) + ((((r34_z / $(32768)) % $(2)) * $(32768)) + (((r34_z / $(256)) % $(128)) * $(256)) + (r34_z % $(256))), dtype=tl.float32)
    r34_value = r34_value.to(tl.bfloat16)
    r34_value = tl.broadcast_to(r34_value, [$(65536)])
    tl.store($(candidate_vto_copy_19_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(65536))
    r35_value = (tl.load($(candidate_vmm_1_ptr) + ((((r35_z / $(32768)) % $(2)) * $(262144)) + (((r35_z / $(32)) % $(128)) * $(2048)) + (((r35_z / $(4096)) % $(8)) * $(256)) + ((r35_z % $(32)) * $(2)) + $(192)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r35_value = r35_value.to(tl.bfloat16)
    r35_value = tl.broadcast_to(r35_value, [$(65536)])
    tl.store($(candidate_vmul_11_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(65536))
    r36_value = (tl.load($(candidate_vmm_1_ptr) + ((((r36_z / $(32768)) % $(2)) * $(262144)) + (((r36_z / $(32)) % $(128)) * $(2048)) + (((r36_z / $(4096)) % $(8)) * $(256)) + ((r36_z % $(32)) * $(2)) + $(193)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(65536)])
    tl.store($(candidate_vmul_12_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(65536))
    r37_value = (tl.load($(candidate_vmul_11_ptr) + ((((r37_z / $(32768)) % $(2)) * $(32768)) + (((r37_z / $(4096)) % $(8)) * $(4096)) + (((r37_z / $(32)) % $(128)) * $(32)) + (r37_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_12_ptr) + ((((r37_z / $(32768)) % $(2)) * $(32768)) + (((r37_z / $(4096)) % $(8)) * $(4096)) + (((r37_z / $(32)) % $(128)) * $(32)) + (r37_z % $(32))), dtype=tl.bfloat16))
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(65536)])
    tl.store($(candidate_vsub_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(65536))
    r38_value = (tl.load($(candidate_vmm_1_ptr) + ((((r38_z / $(32768)) % $(2)) * $(262144)) + (((r38_z / $(32)) % $(128)) * $(2048)) + (((r38_z / $(4096)) % $(8)) * $(256)) + ((r38_z % $(32)) * $(2)) + $(193)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(65536)])
    tl.store($(candidate_vmul_13_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(65536))
    r39_value = (tl.load($(candidate_vmm_1_ptr) + ((((r39_z / $(32768)) % $(2)) * $(262144)) + (((r39_z / $(32)) % $(128)) * $(2048)) + (((r39_z / $(4096)) % $(8)) * $(256)) + ((r39_z % $(32)) * $(2)) + $(192)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(65536)])
    tl.store($(candidate_vmul_14_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(65536))
    r40_value = (tl.load($(candidate_vmul_13_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(4096)) % $(8)) * $(4096)) + (((r40_z / $(32)) % $(128)) * $(32)) + (r40_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_14_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(4096)) % $(8)) * $(4096)) + (((r40_z / $(32)) % $(128)) * $(32)) + (r40_z % $(32))), dtype=tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(candidate_vadd_4_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(131072))
    r41_value = tl.where((r41_z % $(64)) < $(32), tl.load($(candidate_vsub_ptr) + ((((r41_z / $(65536)) % $(2)) * $(32768)) + (((r41_z / $(8192)) % $(8)) * $(4096)) + (((r41_z / $(64)) % $(128)) * $(32)) + (((r41_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_4_ptr) + ((((r41_z / $(65536)) % $(2)) * $(32768)) + (((r41_z / $(8192)) % $(8)) * $(4096)) + (((r41_z / $(64)) % $(128)) * $(32)) + (((r41_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(131072)])
    tl.store($(candidate_vcat_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(8192))
    r42_value = (tl.load($(candidate_vmm_2_ptr) + ((((r42_z / $(4096)) % $(2)) * $(40960)) + (((r42_z / $(32)) % $(128)) * $(320)) + ((r42_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(8192)])
    tl.store($(candidate_vmul_15_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(8192))
    r43_value = (tl.load($(candidate_vmm_2_ptr) + ((((r43_z / $(4096)) % $(2)) * $(40960)) + (((r43_z / $(32)) % $(128)) * $(320)) + ((r43_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(8192)])
    tl.store($(candidate_vmul_16_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(8192))
    r44_value = (tl.load($(candidate_vmul_15_ptr) + ((((r44_z / $(4096)) % $(2)) * $(4096)) + (((r44_z / $(32)) % $(128)) * $(32)) + (r44_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_16_ptr) + ((((r44_z / $(4096)) % $(2)) * $(4096)) + (((r44_z / $(32)) % $(128)) * $(32)) + (r44_z % $(32))), dtype=tl.bfloat16))
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(8192)])
    tl.store($(candidate_vsub_1_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(8192))
    r45_value = (tl.load($(candidate_vmm_2_ptr) + ((((r45_z / $(4096)) % $(2)) * $(40960)) + (((r45_z / $(32)) % $(128)) * $(320)) + ((r45_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(8192)])
    tl.store($(candidate_vmul_17_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(8192))
    r46_value = (tl.load($(candidate_vmm_2_ptr) + ((((r46_z / $(4096)) % $(2)) * $(40960)) + (((r46_z / $(32)) % $(128)) * $(320)) + ((r46_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(8192)])
    tl.store($(candidate_vmul_18_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(8192))
    r47_value = (tl.load($(candidate_vmul_17_ptr) + ((((r47_z / $(4096)) % $(2)) * $(4096)) + (((r47_z / $(32)) % $(128)) * $(32)) + (r47_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_18_ptr) + ((((r47_z / $(4096)) % $(2)) * $(4096)) + (((r47_z / $(32)) % $(128)) * $(32)) + (r47_z % $(32))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(8192)])
    tl.store($(candidate_vadd_5_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(16384))
    r48_value = tl.where((r48_z % $(64)) < $(32), tl.load($(candidate_vsub_1_ptr) + ((((r48_z / $(8192)) % $(2)) * $(4096)) + (((r48_z / $(64)) % $(128)) * $(32)) + (((r48_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_5_ptr) + ((((r48_z / $(8192)) % $(2)) * $(4096)) + (((r48_z / $(64)) % $(128)) * $(32)) + (((r48_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(16384)])
    tl.store($(candidate_vcat_1_ptr) + r48_z, r48_value)
    r49_row = tl.arange($(256))
    r49_col = tl.arange($(1536))
    r49_lane = tl.arange($(256))
    for r49_group in range($(1)) {
      r49_left = tl.load($(candidate_vto_copy_19_ptr) + ((((((((((r49_row[:, None] * $(256)) + r49_lane[None, :]) / $(256)) % $(256)) * $(256)) + (((r49_row[:, None] * $(256)) + r49_lane[None, :]) % $(256))) / $(32768)) % $(2)) * $(32768)) + ((((((r49_row[:, None] * $(256)) + r49_lane[None, :]) / $(256)) % $(256)) % $(128)) * $(256)) + (((r49_row[:, None] * $(256)) + r49_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r49_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r49_lane[:, None] * $(1536)) + r49_col[None, :]) / $(192)) % $(8)) * $(114688)) + ((((r49_lane[:, None] * $(1536)) + r49_col[None, :]) % $(192)) * $(256)) + ((((r49_lane[:, None] * $(1536)) + r49_col[None, :]) / $(1536)) % $(256))), dtype=tl.bfloat16)
      r49_value = tl.dot(r49_left, r49_right)
      r49_value = r49_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r49_group * $(393216)) + (r49_row[:, None] * $(1536)) + r49_col[None, :]), r49_value)
    }
    r50_row = tl.arange($(256))
    r50_col = tl.arange($(2048))
    r50_lane = tl.arange($(256))
    for r50_group in range($(1)) {
      r50_left = tl.load($(candidate_vto_copy_19_ptr) + ((((((((((r50_row[:, None] * $(256)) + r50_lane[None, :]) / $(256)) % $(256)) * $(256)) + (((r50_row[:, None] * $(256)) + r50_lane[None, :]) % $(256))) / $(32768)) % $(2)) * $(32768)) + ((((((r50_row[:, None] * $(256)) + r50_lane[None, :]) / $(256)) % $(256)) % $(128)) * $(256)) + (((r50_row[:, None] * $(256)) + r50_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r50_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r50_lane[:, None] * $(2048)) + r50_col[None, :]) / $(256)) % $(8)) * $(114688)) + ((((r50_lane[:, None] * $(2048)) + r50_col[None, :]) % $(256)) * $(256)) + ((((r50_lane[:, None] * $(2048)) + r50_col[None, :]) / $(2048)) % $(256)) + $(49152)), dtype=tl.bfloat16)
      r50_value = tl.dot(r50_left, r50_right)
      r50_value = r50_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_1_ptr) + ((r50_group * $(524288)) + (r50_row[:, None] * $(2048)) + r50_col[None, :]), r50_value)
    }
    r51_z = tl.arange($(524288))
    r51_value = tl.where((r51_z % $(256)) < $(192), tl.load($(candidate_vmm_1_ptr) + ((((((((r51_z / $(2048)) % $(256)) * $(1536)) + (((r51_z / $(256)) % $(8)) * $(192)) + ((r51_z % $(256)) - $(0))) / $(196608)) % $(2)) * $(262144)) + (((((((r51_z / $(2048)) % $(256)) * $(1536)) + (((r51_z / $(256)) % $(8)) * $(192)) + ((r51_z % $(256)) - $(0))) / $(1536)) % $(128)) * $(2048)) + (((((((r51_z / $(2048)) % $(256)) * $(1536)) + (((r51_z / $(256)) % $(8)) * $(192)) + ((r51_z % $(256)) - $(0))) / $(192)) % $(8)) * $(256)) + (((((r51_z / $(2048)) % $(256)) * $(1536)) + (((r51_z / $(256)) % $(8)) * $(192)) + ((r51_z % $(256)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_ptr) + ((((((((r51_z / $(2048)) % $(256)) * $(512)) + (((r51_z / $(256)) % $(8)) * $(64)) + ((r51_z % $(256)) - $(192))) / $(65536)) % $(2)) * $(65536)) + (((((((r51_z / $(2048)) % $(256)) * $(512)) + (((r51_z / $(256)) % $(8)) * $(64)) + ((r51_z % $(256)) - $(192))) / $(64)) % $(8)) * $(8192)) + (((((((r51_z / $(2048)) % $(256)) * $(512)) + (((r51_z / $(256)) % $(8)) * $(64)) + ((r51_z % $(256)) - $(192))) / $(512)) % $(128)) * $(64)) + (((((r51_z / $(2048)) % $(256)) * $(512)) + (((r51_z / $(256)) % $(8)) * $(64)) + ((r51_z % $(256)) - $(192))) % $(64))), dtype=tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(524288)])
    tl.store($(candidate_vcat_2_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(524288))
    r52_value = tl.where((r52_z % $(256)) < $(192), tl.load($(candidate_vbmm_ptr) + ((((((((r52_z / $(2048)) % $(256)) * $(1536)) + (((r52_z / $(256)) % $(8)) * $(192)) + ((r52_z % $(256)) - $(0))) / $(1536)) % $(256)) * $(1536)) + (((((((r52_z / $(2048)) % $(256)) * $(1536)) + (((r52_z / $(256)) % $(8)) * $(192)) + ((r52_z % $(256)) - $(0))) / $(192)) % $(8)) * $(192)) + (((((r52_z / $(2048)) % $(256)) * $(1536)) + (((r52_z / $(256)) % $(8)) * $(192)) + ((r52_z % $(256)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_1_ptr) + ((((((((r52_z / $(2048)) % $(256)) * $(64)) + ((r52_z % $(256)) - $(192))) / $(8192)) % $(2)) * $(8192)) + (((((((r52_z / $(2048)) % $(256)) * $(64)) + ((r52_z % $(256)) - $(192))) / $(64)) % $(128)) * $(64)) + (((((r52_z / $(2048)) % $(256)) * $(64)) + ((r52_z % $(256)) - $(192))) % $(64))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(524288)])
    tl.store($(candidate_vcat_3_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(262144))
    r53_value = tl.load($(candidate_vcat_2_ptr) + ((((r53_z / $(256)) % $(128)) * $(2048)) + (((r53_z / $(32768)) % $(8)) * $(256)) + (r53_z % $(256))), dtype=tl.bfloat16)
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(262144)])
    tl.store($(candidate_vto_copy_20_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(262144))
    r54_value = tl.load($(candidate_vcat_3_ptr) + ((((r54_z / $(256)) % $(128)) * $(2048)) + (((r54_z / $(32768)) % $(8)) * $(256)) + (r54_z % $(256))), dtype=tl.bfloat16)
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(262144)])
    tl.store($(candidate_vto_copy_21_ptr) + r54_z, r54_value)
    r55_row = tl.arange($(128))
    r55_col = tl.arange($(128))
    r55_lane = tl.arange($(256))
    for r55_group in range($(8)) {
      r55_left = tl.load($(candidate_vto_copy_20_ptr) + (((((((r55_group % $(8)) * $(32768)) + (r55_row[:, None] * $(256)) + r55_lane[None, :]) / $(32768)) % $(8)) * $(32768)) + ((((((r55_group % $(8)) * $(32768)) + (r55_row[:, None] * $(256)) + r55_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r55_group % $(8)) * $(32768)) + (r55_row[:, None] * $(256)) + r55_lane[None, :]) % $(256))), dtype=tl.float32)
      r55_right = tl.load($(candidate_vto_copy_21_ptr) + (((((((r55_group % $(8)) * $(32768)) + (r55_lane[:, None] * $(128)) + r55_col[None, :]) / $(32768)) % $(8)) * $(32768)) + (((((r55_group % $(8)) * $(32768)) + (r55_lane[:, None] * $(128)) + r55_col[None, :]) % $(128)) * $(256)) + (((((r55_group % $(8)) * $(32768)) + (r55_lane[:, None] * $(128)) + r55_col[None, :]) / $(128)) % $(256))), dtype=tl.float32)
      r55_value = tl.dot(r55_left, r55_right)
      r55_value = r55_value.to(tl.float32)
      tl.store($(candidate_vbmm_2_ptr) + ((r55_group * $(16384)) + (r55_row[:, None] * $(128)) + r55_col[None, :]), r55_value)
    }
    r56_z = tl.arange($(131072))
    r56_value = (tl.load($(candidate_vbmm_2_ptr) + ((((r56_z / $(16384)) % $(8)) * $(16384)) + (((r56_z / $(128)) % $(128)) * $(128)) + (r56_z % $(128))), dtype=tl.float32) * 0.0625)
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(131072)])
    tl.store($(candidate_vmul_19_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(128))
    r57_value = (tl.cast(r57_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r57_value = tl.broadcast_to(r57_value, [$(128)])
    tl.store($(candidate_vadd_6_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(128))
    r58_value = (tl.load($(candidate_vadd_6_ptr) + r58_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r58_value = tl.broadcast_to(r58_value, [$(128)])
    tl.store($(candidate_vsub_2_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(16384))
    r59_value = (tl.cast((r59_z % $(128)), tl.int64) > tl.load($(candidate_vsub_2_ptr) + ((r59_z / $(128)) % $(128)), dtype=tl.int64))
    r59_value = tl.broadcast_to(r59_value, [$(16384)])
    tl.store($(candidate_vgt_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(131072))
    r60_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r60_z / $(128)) % $(128)) * $(128)) + (r60_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_19_ptr) + ((((r60_z / $(16384)) % $(8)) * $(16384)) + (((r60_z / $(128)) % $(128)) * $(128)) + (r60_z % $(128))), dtype=tl.float32))
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(131072))
    r61_lane = tl.arange($(128))
    r61_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r61_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r61_z[:, None] / $(128)) % $(128)) * $(128)) + (r61_lane[None, :] % $(128))), dtype=tl.float32)
    r61_maximum = tl.max(r61_row_values, axis=1, keep_dims=true)
    r61_weights = tl.exp(r61_row_values - r61_maximum)
    r61_total = tl.sum(r61_weights, axis=1)
    r61_maximum = tl.max(r61_row_values, axis=1)
    r61_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r61_z / $(16384)) % $(8)) * $(16384)) + (((r61_z / $(128)) % $(128)) * $(128)) + (r61_z % $(128))), dtype=tl.float32) - r61_maximum) / r61_total
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(131072)])
    tl.store($(candidate_vsoftmax_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(131072))
    r62_value = tl.load($(candidate_vsoftmax_ptr) + ((((r62_z / $(16384)) % $(8)) * $(16384)) + (((r62_z / $(128)) % $(128)) * $(128)) + (r62_z % $(128))), dtype=tl.float32)
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(131072)])
    tl.store($(candidate_vto_copy_22_ptr) + r62_z, r62_value)
    r63_row = tl.arange($(128))
    r63_col = tl.arange($(256))
    r63_lane = tl.arange($(128))
    for r63_group in range($(8)) {
      r63_left = tl.load($(candidate_vto_copy_22_ptr) + (((((((r63_group % $(8)) * $(16384)) + (r63_row[:, None] * $(128)) + r63_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r63_group % $(8)) * $(16384)) + (r63_row[:, None] * $(128)) + r63_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r63_group % $(8)) * $(16384)) + (r63_row[:, None] * $(128)) + r63_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r63_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r63_group % $(8)) * $(32768)) + (r63_lane[:, None] * $(256)) + r63_col[None, :]) / $(256)) % $(128)) * $(2048)) + ((((((r63_group % $(8)) * $(32768)) + (r63_lane[:, None] * $(256)) + r63_col[None, :]) / $(32768)) % $(8)) * $(256)) + ((((r63_group % $(8)) * $(32768)) + (r63_lane[:, None] * $(256)) + r63_col[None, :]) % $(256))), dtype=tl.bfloat16)
      r63_value = tl.dot(r63_left, r63_right)
      r63_value = r63_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_3_ptr) + ((r63_group * $(32768)) + (r63_row[:, None] * $(256)) + r63_col[None, :]), r63_value)
    }
    r64_z = tl.arange($(262144))
    r64_value = tl.load($(candidate_vcat_2_ptr) + ((((r64_z / $(256)) % $(128)) * $(2048)) + (((r64_z / $(32768)) % $(8)) * $(256)) + (r64_z % $(256)) + $(262144)), dtype=tl.bfloat16)
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(262144)])
    tl.store($(candidate_vto_copy_23_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(262144))
    r65_value = tl.load($(candidate_vcat_3_ptr) + ((((r65_z / $(256)) % $(128)) * $(2048)) + (((r65_z / $(32768)) % $(8)) * $(256)) + (r65_z % $(256)) + $(262144)), dtype=tl.bfloat16)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(262144)])
    tl.store($(candidate_vto_copy_24_ptr) + r65_z, r65_value)
    r66_row = tl.arange($(128))
    r66_col = tl.arange($(128))
    r66_lane = tl.arange($(256))
    for r66_group in range($(8)) {
      r66_left = tl.load($(candidate_vto_copy_23_ptr) + (((((((r66_group % $(8)) * $(32768)) + (r66_row[:, None] * $(256)) + r66_lane[None, :]) / $(32768)) % $(8)) * $(32768)) + ((((((r66_group % $(8)) * $(32768)) + (r66_row[:, None] * $(256)) + r66_lane[None, :]) / $(256)) % $(128)) * $(256)) + ((((r66_group % $(8)) * $(32768)) + (r66_row[:, None] * $(256)) + r66_lane[None, :]) % $(256))), dtype=tl.float32)
      r66_right = tl.load($(candidate_vto_copy_24_ptr) + (((((((r66_group % $(8)) * $(32768)) + (r66_lane[:, None] * $(128)) + r66_col[None, :]) / $(32768)) % $(8)) * $(32768)) + (((((r66_group % $(8)) * $(32768)) + (r66_lane[:, None] * $(128)) + r66_col[None, :]) % $(128)) * $(256)) + (((((r66_group % $(8)) * $(32768)) + (r66_lane[:, None] * $(128)) + r66_col[None, :]) / $(128)) % $(256))), dtype=tl.float32)
      r66_value = tl.dot(r66_left, r66_right)
      r66_value = r66_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r66_group * $(16384)) + (r66_row[:, None] * $(128)) + r66_col[None, :]), r66_value)
    }
    r67_z = tl.arange($(131072))
    r67_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r67_z / $(16384)) % $(8)) * $(16384)) + (((r67_z / $(128)) % $(128)) * $(128)) + (r67_z % $(128))), dtype=tl.float32) * 0.0625)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(131072)])
    tl.store($(candidate_vmul_20_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(128))
    r68_value = (tl.cast(r68_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r68_value = tl.broadcast_to(r68_value, [$(128)])
    tl.store($(candidate_vadd_7_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(128))
    r69_value = (tl.load($(candidate_vadd_7_ptr) + r69_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r69_value = tl.broadcast_to(r69_value, [$(128)])
    tl.store($(candidate_vsub_3_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(16384))
    r70_value = (tl.cast((r70_z % $(128)), tl.int64) > tl.load($(candidate_vsub_3_ptr) + ((r70_z / $(128)) % $(128)), dtype=tl.int64))
    r70_value = tl.broadcast_to(r70_value, [$(16384)])
    tl.store($(candidate_vgt_1_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(131072))
    r71_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r71_z / $(128)) % $(128)) * $(128)) + (r71_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_20_ptr) + ((((r71_z / $(16384)) % $(8)) * $(16384)) + (((r71_z / $(128)) % $(128)) * $(128)) + (r71_z % $(128))), dtype=tl.float32))
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(131072))
    r72_lane = tl.arange($(128))
    r72_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r72_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r72_z[:, None] / $(128)) % $(128)) * $(128)) + (r72_lane[None, :] % $(128))), dtype=tl.float32)
    r72_maximum = tl.max(r72_row_values, axis=1, keep_dims=true)
    r72_weights = tl.exp(r72_row_values - r72_maximum)
    r72_total = tl.sum(r72_weights, axis=1)
    r72_maximum = tl.max(r72_row_values, axis=1)
    r72_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r72_z / $(16384)) % $(8)) * $(16384)) + (((r72_z / $(128)) % $(128)) * $(128)) + (r72_z % $(128))), dtype=tl.float32) - r72_maximum) / r72_total
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(131072)])
    tl.store($(candidate_vsoftmax_1_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(131072))
    r73_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r73_z / $(16384)) % $(8)) * $(16384)) + (((r73_z / $(128)) % $(128)) * $(128)) + (r73_z % $(128))), dtype=tl.float32)
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(131072)])
    tl.store($(candidate_vto_copy_25_ptr) + r73_z, r73_value)
    r74_row = tl.arange($(128))
    r74_col = tl.arange($(256))
    r74_lane = tl.arange($(128))
    for r74_group in range($(8)) {
      r74_left = tl.load($(candidate_vto_copy_25_ptr) + (((((((r74_group % $(8)) * $(16384)) + (r74_row[:, None] * $(128)) + r74_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r74_group % $(8)) * $(16384)) + (r74_row[:, None] * $(128)) + r74_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r74_group % $(8)) * $(16384)) + (r74_row[:, None] * $(128)) + r74_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r74_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r74_group % $(8)) * $(32768)) + (r74_lane[:, None] * $(256)) + r74_col[None, :]) / $(256)) % $(128)) * $(2048)) + ((((((r74_group % $(8)) * $(32768)) + (r74_lane[:, None] * $(256)) + r74_col[None, :]) / $(32768)) % $(8)) * $(256)) + ((((r74_group % $(8)) * $(32768)) + (r74_lane[:, None] * $(256)) + r74_col[None, :]) % $(256)) + $(262144)), dtype=tl.bfloat16)
      r74_value = tl.dot(r74_left, r74_right)
      r74_value = r74_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r74_group * $(32768)) + (r74_row[:, None] * $(256)) + r74_col[None, :]), r74_value)
    }
    r75_z = tl.arange($(524288))
    r75_value = tl.where(((r75_z / $(2048)) % $(256)) < $(128), tl.load($(candidate_vbmm_3_ptr) + ((((((((r75_z / $(256)) % $(8)) * $(32768)) + ((((r75_z / $(2048)) % $(256)) - $(0)) * $(256)) + (r75_z % $(256))) / $(32768)) % $(8)) * $(32768)) + ((((((r75_z / $(256)) % $(8)) * $(128)) + (((r75_z / $(2048)) % $(256)) - $(0))) % $(128)) * $(256)) + (r75_z % $(256))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_5_ptr) + ((((((((r75_z / $(256)) % $(8)) * $(32768)) + ((((r75_z / $(2048)) % $(256)) - $(128)) * $(256)) + (r75_z % $(256))) / $(32768)) % $(8)) * $(32768)) + ((((((r75_z / $(256)) % $(8)) * $(128)) + (((r75_z / $(2048)) % $(256)) - $(128))) % $(128)) * $(256)) + (r75_z % $(256))), dtype=tl.bfloat16))
    r75_value = r75_value.to(tl.bfloat16)
    r75_value = tl.broadcast_to(r75_value, [$(524288)])
    tl.store($(candidate_vcat_4_ptr) + r75_z, r75_value)
    r76_row = tl.arange($(256))
    r76_col = tl.arange($(512))
    r76_lane = tl.arange($(2048))
    r76_left = tl.load($(candidate_vcat_4_ptr) + ((((((r76_row[:, None] * $(2048)) + r76_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r76_row[:, None] * $(2048)) + r76_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((((((((r76_row[:, None] * $(2048)) + r76_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r76_row[:, None] * $(2048)) + r76_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r76_row[:, None] * $(2048)) + r76_lane[None, :]) % $(2048))) / $(256)) % $(8)) * $(256)) + (((((((r76_row[:, None] * $(2048)) + r76_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r76_row[:, None] * $(2048)) + r76_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r76_row[:, None] * $(2048)) + r76_lane[None, :]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r76_right = tl.load($(candidate_vto_copy_6_ptr) + (((r76_col[None, :] % $(512)) * $(2048)) + (r76_lane[:, None] % $(2048))), dtype=tl.bfloat16)
    r76_value = tl.dot(r76_left, r76_right)
    r76_value = r76_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_3_ptr) + ((r76_row[:, None] * $(512)) + r76_col[None, :]), r76_value)
    z = tl.arange($(131072))
    value = tl.load($(candidate_vmm_3_ptr) + ((((z / $(65536)) % $(2)) * $(65536)) + (((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_nonabsorb_fwd
