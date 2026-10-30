import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_absorb_fwd
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
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vbmm_ptr : RegionName := "vbmm"
  let vcat_ptr : RegionName := "vcat"
  let vcos_ptr : RegionName := "vcos"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vsin_ptr : RegionName := "vsin"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_ptr : RegionName := "vsub"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vbmm_3_ptr : RegionName := "vbmm_3"
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
    r16_z = tl.arange($(32))
    r16_value = tl.toReal((tl.cast(r16_z, tl.int64) * tl.full([], $(2), dtype=tl.int64)))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(32)])
    tl.store($(vto_copy_14_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32))
    r17_value = (tl.load($(vto_copy_14_ptr) + r17_z, dtype=tl.float32) / 64.0)
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
    r21_z = tl.arange($(128))
    r21_value = tl.toReal(tl.cast(r21_z, tl.int64))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(128)])
    tl.store($(vto_copy_16_ptr) + r21_z, r21_value)
    r22_row = tl.arange($(32))
    r22_col = tl.arange($(128))
    r22_lane = tl.arange($(1))
    for r22_group in range($(1)) {
      r22_left = tl.load($(vmul_7_ptr) + ((r22_row[:, None] + r22_lane[None, :]) % $(32)), dtype=tl.float32)
      r22_right = tl.load($(vto_copy_16_ptr) + (((r22_lane[:, None] * $(128)) + r22_col[None, :]) % $(128)), dtype=tl.float32)
      r22_value = tl.dot(r22_left, r22_right)
      r22_value = r22_value.to(tl.float32)
      tl.store($(vbmm_ptr) + ((r22_group * $(4096)) + (r22_row[:, None] * $(128)) + r22_col[None, :]), r22_value)
    }
    r23_z = tl.arange($(8192))
    r23_value = tl.where((r23_z % $(64)) < $(32), tl.load($(vbmm_ptr) + (((((r23_z % $(64)) - $(0)) % $(32)) * $(128)) + ((r23_z / $(64)) % $(128))), dtype=tl.float32), tl.load($(vbmm_ptr) + (((((r23_z % $(64)) - $(32)) % $(32)) * $(128)) + ((r23_z / $(64)) % $(128))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(8192)])
    tl.store($(vcat_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(8192))
    r24_value = tl.cos(tl.load($(vcat_ptr) + ((((r24_z / $(64)) % $(128)) * $(64)) + (r24_z % $(64))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(8192)])
    tl.store($(vcos_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(8192))
    r25_value = (tl.load($(vcos_ptr) + ((((r25_z / $(64)) % $(128)) * $(64)) + (r25_z % $(64))), dtype=tl.float32) * 1.0)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(8192)])
    tl.store($(vmul_9_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(8192))
    r26_value = tl.sin(tl.load($(vcat_ptr) + ((((r26_z / $(64)) % $(128)) * $(64)) + (r26_z % $(64))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(8192)])
    tl.store($(vsin_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(8192))
    r27_value = (tl.load($(vsin_ptr) + ((((r27_z / $(64)) % $(128)) * $(64)) + (r27_z % $(64))), dtype=tl.float32) * 1.0)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(8192)])
    tl.store($(vmul_10_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(8192))
    r28_value = tl.load($(vmul_9_ptr) + ((((r28_z / $(64)) % $(128)) * $(64)) + (r28_z % $(64))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(8192)])
    tl.store($(vto_copy_17_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(8192))
    r29_value = tl.load($(vmul_10_ptr) + ((((r29_z / $(64)) % $(128)) * $(64)) + (r29_z % $(64))), dtype=tl.float32)
    r29_value = r29_value.to(tl.bfloat16)
    r29_value = tl.broadcast_to(r29_value, [$(8192)])
    tl.store($(vto_copy_18_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(16384))
    r30_value = tl.where((r30_z % $(128)) >= ((r30_z / $(128)) % $(128)) + $(1), true, false)
    r30_value = tl.broadcast_to(r30_value, [$(16384)])
    tl.store($(vtriu_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(32768))
    r31_value = tl.where(tl.load($(vtriu_ptr) + ((((r31_z / $(128)) % $(128)) * $(128)) + (r31_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r31_value = r31_value.to(tl.bfloat16)
    r31_value = tl.broadcast_to(r31_value, [$(32768)])
    tl.store($(vmasked_fill_ptr) + r31_z, r31_value)
    r32_row = tl.arange($(256))
    r32_col = tl.arange($(256))
    r32_lane = tl.arange($(512))
    r32_left = tl.load($(input0_ptr) + ((((((r32_row[:, None] * $(512)) + r32_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r32_row[:, None] * $(512)) + r32_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r32_row[:, None] * $(512)) + r32_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r32_right = tl.load($(vto_copy_ptr) + (((r32_col[None, :] % $(256)) * $(512)) + (r32_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r32_value = tl.dot(r32_left, r32_right)
    r32_value = r32_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r32_row[:, None] * $(256)) + r32_col[None, :]), r32_value)
    r33_z = tl.arange($(65536))
    r33_value = tl.load($(vmm_ptr) + ((((r33_z / $(32768)) % $(2)) * $(32768)) + (((r33_z / $(256)) % $(128)) * $(256)) + (r33_z % $(256))), dtype=tl.bfloat16)
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(65536)])
    tl.store($(vto_copy_20_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(65536))
    r34_value = ((tl.load($(vto_copy_20_ptr) + ((((r34_z / $(32768)) % $(2)) * $(32768)) + (((r34_z / $(256)) % $(128)) * $(256)) + (r34_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_20_ptr) + ((((r34_z / $(32768)) % $(2)) * $(32768)) + (((r34_z / $(256)) % $(128)) * $(256)) + (r34_z % $(256))), dtype=tl.float32)))
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(65536)])
    tl.store($(vpow_3_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(256))
    r35_lane = tl.arange($(256))
    r35_value = (tl.sum(tl.load($(vpow_3_ptr) + ((((r35_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r35_z[:, None] % $(128)) * $(256)) + r35_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(256)])
    tl.store($(vmean_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(256))
    r36_value = (tl.load($(vmean_ptr) + ((((r36_z / $(128)) % $(2)) * $(128)) + (r36_z % $(128))), dtype=tl.float32) + 0.000001)
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(256)])
    tl.store($(vadd_2_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(256))
    r37_value = tl.rsqrt(tl.load($(vadd_2_ptr) + ((((r37_z / $(128)) % $(2)) * $(128)) + (r37_z % $(128))), dtype=tl.float32))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(256)])
    tl.store($(vrsqrt_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(65536))
    r38_value = (tl.load($(vto_copy_20_ptr) + ((((r38_z / $(32768)) % $(2)) * $(32768)) + (((r38_z / $(256)) % $(128)) * $(256)) + (r38_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r38_z / $(32768)) % $(2)) * $(128)) + ((r38_z / $(256)) % $(128))), dtype=tl.float32))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(65536)])
    tl.store($(vmul_13_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(65536))
    r39_value = (tl.cast(tl.load($(vto_copy_1_ptr) + (r39_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_13_ptr) + ((((r39_z / $(32768)) % $(2)) * $(32768)) + (((r39_z / $(256)) % $(128)) * $(256)) + (r39_z % $(256))), dtype=tl.float32))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(65536)])
    tl.store($(vmul_14_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(65536))
    r40_value = tl.load($(vmul_14_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(256)) % $(128)) * $(256)) + (r40_z % $(256))), dtype=tl.float32)
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(vto_copy_21_ptr) + r40_z, r40_value)
    r41_row = tl.arange($(256))
    r41_col = tl.arange($(1536))
    r41_lane = tl.arange($(256))
    r41_left = tl.load($(vto_copy_21_ptr) + ((((((r41_row[:, None] * $(256)) + r41_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r41_row[:, None] * $(256)) + r41_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r41_row[:, None] * $(256)) + r41_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r41_right = tl.load($(vto_copy_2_ptr) + (((r41_col[None, :] % $(1536)) * $(256)) + (r41_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r41_value = tl.dot(r41_left, r41_right)
    r41_value = r41_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r41_row[:, None] * $(1536)) + r41_col[None, :]), r41_value)
    r42_row = tl.arange($(256))
    r42_col = tl.arange($(576))
    r42_lane = tl.arange($(512))
    r42_left = tl.load($(input0_ptr) + ((((((r42_row[:, None] * $(512)) + r42_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r42_row[:, None] * $(512)) + r42_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r42_row[:, None] * $(512)) + r42_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r42_right = tl.load($(vto_copy_3_ptr) + (((r42_col[None, :] % $(576)) * $(512)) + (r42_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r42_value = tl.dot(r42_left, r42_right)
    r42_value = r42_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r42_row[:, None] * $(576)) + r42_col[None, :]), r42_value)
    r43_z = tl.arange($(131072))
    r43_value = tl.load($(vmm_2_ptr) + ((((r43_z / $(65536)) % $(2)) * $(73728)) + (((r43_z / $(512)) % $(128)) * $(576)) + (r43_z % $(512))), dtype=tl.bfloat16)
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(131072)])
    tl.store($(vto_copy_22_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(131072))
    r44_value = ((tl.load($(vto_copy_22_ptr) + ((((r44_z / $(65536)) % $(2)) * $(65536)) + (((r44_z / $(512)) % $(128)) * $(512)) + (r44_z % $(512))), dtype=tl.float32)) * (tl.load($(vto_copy_22_ptr) + ((((r44_z / $(65536)) % $(2)) * $(65536)) + (((r44_z / $(512)) % $(128)) * $(512)) + (r44_z % $(512))), dtype=tl.float32)))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(131072)])
    tl.store($(vpow_4_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(256))
    r45_lane = tl.arange($(512))
    r45_value = (tl.sum(tl.load($(vpow_4_ptr) + ((((r45_z[:, None] / $(128)) % $(2)) * $(65536)) + ((r45_z[:, None] % $(128)) * $(512)) + r45_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(256)])
    tl.store($(vmean_1_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(256))
    r46_value = (tl.load($(vmean_1_ptr) + ((((r46_z / $(128)) % $(2)) * $(128)) + (r46_z % $(128))), dtype=tl.float32) + 0.000001)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(256)])
    tl.store($(vadd_3_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(256))
    r47_value = tl.rsqrt(tl.load($(vadd_3_ptr) + ((((r47_z / $(128)) % $(2)) * $(128)) + (r47_z % $(128))), dtype=tl.float32))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(256)])
    tl.store($(vrsqrt_1_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(131072))
    r48_value = (tl.load($(vto_copy_22_ptr) + ((((r48_z / $(65536)) % $(2)) * $(65536)) + (((r48_z / $(512)) % $(128)) * $(512)) + (r48_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((((r48_z / $(65536)) % $(2)) * $(128)) + ((r48_z / $(512)) % $(128))), dtype=tl.float32))
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(131072)])
    tl.store($(vmul_15_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(131072))
    r49_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r49_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_15_ptr) + ((((r49_z / $(65536)) % $(2)) * $(65536)) + (((r49_z / $(512)) % $(128)) * $(512)) + (r49_z % $(512))), dtype=tl.float32))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(131072)])
    tl.store($(vmul_16_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(131072))
    r50_value = tl.load($(vmul_16_ptr) + ((((r50_z / $(65536)) % $(2)) * $(65536)) + (((r50_z / $(512)) % $(128)) * $(512)) + (r50_z % $(512))), dtype=tl.float32)
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(131072)])
    tl.store($(vto_copy_23_ptr) + r50_z, r50_value)
    r51_row = tl.arange($(256))
    r51_col = tl.arange($(2048))
    r51_lane = tl.arange($(512))
    r51_left = tl.load($(vto_copy_23_ptr) + ((((((r51_row[:, None] * $(512)) + r51_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r51_row[:, None] * $(512)) + r51_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r51_row[:, None] * $(512)) + r51_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r51_right = tl.load($(vto_copy_5_ptr) + (((r51_col[None, :] % $(2048)) * $(512)) + (r51_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r51_value = tl.dot(r51_left, r51_right)
    r51_value = r51_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r51_row[:, None] * $(2048)) + r51_col[None, :]), r51_value)
    r52_z = tl.arange($(65536))
    r52_value = (tl.load($(vmm_1_ptr) + ((((r52_z / $(32768)) % $(2)) * $(196608)) + (((r52_z / $(32)) % $(128)) * $(1536)) + (((r52_z / $(4096)) % $(8)) * $(192)) + ((r52_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r52_z / $(32)) % $(128)) * $(64)) + (r52_z % $(32))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(65536)])
    tl.store($(vmul_17_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(65536))
    r53_value = (tl.load($(vmm_1_ptr) + ((((r53_z / $(32768)) % $(2)) * $(196608)) + (((r53_z / $(32)) % $(128)) * $(1536)) + (((r53_z / $(4096)) % $(8)) * $(192)) + ((r53_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r53_z / $(32)) % $(128)) * $(64)) + (r53_z % $(32))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(65536)])
    tl.store($(vmul_18_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(65536))
    r54_value = (tl.load($(vmul_17_ptr) + ((((r54_z / $(32768)) % $(2)) * $(32768)) + (((r54_z / $(4096)) % $(8)) * $(4096)) + (((r54_z / $(32)) % $(128)) * $(32)) + (r54_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_18_ptr) + ((((r54_z / $(32768)) % $(2)) * $(32768)) + (((r54_z / $(4096)) % $(8)) * $(4096)) + (((r54_z / $(32)) % $(128)) * $(32)) + (r54_z % $(32))), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(65536)])
    tl.store($(vsub_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(65536))
    r55_value = (tl.load($(vmm_1_ptr) + ((((r55_z / $(32768)) % $(2)) * $(196608)) + (((r55_z / $(32)) % $(128)) * $(1536)) + (((r55_z / $(4096)) % $(8)) * $(192)) + ((r55_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r55_z / $(32)) % $(128)) * $(64)) + (r55_z % $(32))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(65536)])
    tl.store($(vmul_19_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(65536))
    r56_value = (tl.load($(vmm_1_ptr) + ((((r56_z / $(32768)) % $(2)) * $(196608)) + (((r56_z / $(32)) % $(128)) * $(1536)) + (((r56_z / $(4096)) % $(8)) * $(192)) + ((r56_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r56_z / $(32)) % $(128)) * $(64)) + (r56_z % $(32))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(65536)])
    tl.store($(vmul_20_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(65536))
    r57_value = (tl.load($(vmul_19_ptr) + ((((r57_z / $(32768)) % $(2)) * $(32768)) + (((r57_z / $(4096)) % $(8)) * $(4096)) + (((r57_z / $(32)) % $(128)) * $(32)) + (r57_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_20_ptr) + ((((r57_z / $(32768)) % $(2)) * $(32768)) + (((r57_z / $(4096)) % $(8)) * $(4096)) + (((r57_z / $(32)) % $(128)) * $(32)) + (r57_z % $(32))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(65536)])
    tl.store($(vadd_4_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(131072))
    r58_value = tl.where((r58_z % $(64)) < $(32), tl.load($(vsub_ptr) + ((((r58_z / $(65536)) % $(2)) * $(32768)) + (((r58_z / $(8192)) % $(8)) * $(4096)) + (((r58_z / $(64)) % $(128)) * $(32)) + (((r58_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_4_ptr) + ((((r58_z / $(65536)) % $(2)) * $(32768)) + (((r58_z / $(8192)) % $(8)) * $(4096)) + (((r58_z / $(64)) % $(128)) * $(32)) + (((r58_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(131072)])
    tl.store($(vcat_2_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(8192))
    r59_value = (tl.load($(vmm_2_ptr) + ((((r59_z / $(4096)) % $(2)) * $(73728)) + (((r59_z / $(32)) % $(128)) * $(576)) + ((r59_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r59_z / $(32)) % $(128)) * $(64)) + (r59_z % $(32))), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(8192)])
    tl.store($(vmul_21_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(8192))
    r60_value = (tl.load($(vmm_2_ptr) + ((((r60_z / $(4096)) % $(2)) * $(73728)) + (((r60_z / $(32)) % $(128)) * $(576)) + ((r60_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r60_z / $(32)) % $(128)) * $(64)) + (r60_z % $(32))), dtype=tl.bfloat16))
    r60_value = r60_value.to(tl.bfloat16)
    r60_value = tl.broadcast_to(r60_value, [$(8192)])
    tl.store($(vmul_22_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(8192))
    r61_value = (tl.load($(vmul_21_ptr) + ((((r61_z / $(4096)) % $(2)) * $(4096)) + (((r61_z / $(32)) % $(128)) * $(32)) + (r61_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_22_ptr) + ((((r61_z / $(4096)) % $(2)) * $(4096)) + (((r61_z / $(32)) % $(128)) * $(32)) + (r61_z % $(32))), dtype=tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(8192)])
    tl.store($(vsub_1_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(8192))
    r62_value = (tl.load($(vmm_2_ptr) + ((((r62_z / $(4096)) % $(2)) * $(73728)) + (((r62_z / $(32)) % $(128)) * $(576)) + ((r62_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r62_z / $(32)) % $(128)) * $(64)) + (r62_z % $(32))), dtype=tl.bfloat16))
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(8192)])
    tl.store($(vmul_23_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(8192))
    r63_value = (tl.load($(vmm_2_ptr) + ((((r63_z / $(4096)) % $(2)) * $(73728)) + (((r63_z / $(32)) % $(128)) * $(576)) + ((r63_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r63_z / $(32)) % $(128)) * $(64)) + (r63_z % $(32))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(8192)])
    tl.store($(vmul_24_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(8192))
    r64_value = (tl.load($(vmul_23_ptr) + ((((r64_z / $(4096)) % $(2)) * $(4096)) + (((r64_z / $(32)) % $(128)) * $(32)) + (r64_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_24_ptr) + ((((r64_z / $(4096)) % $(2)) * $(4096)) + (((r64_z / $(32)) % $(128)) * $(32)) + (r64_z % $(32))), dtype=tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(8192)])
    tl.store($(vadd_5_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(16384))
    r65_value = tl.where((r65_z % $(64)) < $(32), tl.load($(vsub_1_ptr) + ((((r65_z / $(8192)) % $(2)) * $(4096)) + (((r65_z / $(64)) % $(128)) * $(32)) + (((r65_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r65_z / $(8192)) % $(2)) * $(4096)) + (((r65_z / $(64)) % $(128)) * $(32)) + (((r65_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(16384)])
    tl.store($(vcat_3_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(393216))
    r66_value = tl.where((r66_z % $(192)) < $(128), tl.load($(vmm_1_ptr) + ((((((((r66_z / $(196608)) % $(2)) * $(196608)) + (((r66_z / $(192)) % $(128)) * $(1536)) + (((r66_z / $(24576)) % $(8)) * $(192)) + ((r66_z % $(192)) - $(0))) / $(196608)) % $(2)) * $(196608)) + (((((((r66_z / $(196608)) % $(2)) * $(196608)) + (((r66_z / $(192)) % $(128)) * $(1536)) + (((r66_z / $(24576)) % $(8)) * $(192)) + ((r66_z % $(192)) - $(0))) / $(1536)) % $(128)) * $(1536)) + (((((r66_z / $(196608)) % $(2)) * $(196608)) + (((r66_z / $(192)) % $(128)) * $(1536)) + (((r66_z / $(24576)) % $(8)) * $(192)) + ((r66_z % $(192)) - $(0))) % $(1536))), dtype=tl.bfloat16), tl.load($(vcat_2_ptr) + ((((r66_z / $(196608)) % $(2)) * $(65536)) + (((r66_z / $(24576)) % $(8)) * $(8192)) + (((r66_z / $(192)) % $(128)) * $(64)) + (((r66_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(393216)])
    tl.store($(vcat_4_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(393216))
    r67_value = tl.where((r67_z % $(192)) < $(128), tl.load($(vmm_3_ptr) + ((((((((r67_z / $(196608)) % $(2)) * $(262144)) + (((r67_z / $(192)) % $(128)) * $(2048)) + (((r67_z / $(24576)) % $(8)) * $(256)) + ((r67_z % $(192)) - $(0))) / $(262144)) % $(2)) * $(262144)) + (((((((r67_z / $(196608)) % $(2)) * $(262144)) + (((r67_z / $(192)) % $(128)) * $(2048)) + (((r67_z / $(24576)) % $(8)) * $(256)) + ((r67_z % $(192)) - $(0))) / $(2048)) % $(128)) * $(2048)) + (((((r67_z / $(196608)) % $(2)) * $(262144)) + (((r67_z / $(192)) % $(128)) * $(2048)) + (((r67_z / $(24576)) % $(8)) * $(256)) + ((r67_z % $(192)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_3_ptr) + ((((r67_z / $(196608)) % $(2)) * $(8192)) + (((r67_z / $(192)) % $(128)) * $(64)) + (((r67_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(393216)])
    tl.store($(vcat_5_ptr) + r67_z, r67_value)
    r68_row = tl.arange($(128))
    r68_col = tl.arange($(128))
    r68_lane = tl.arange($(192))
    for r68_group in range($(16)) {
      r68_left = tl.load($(vcat_4_ptr) + (((((((r68_group % $(16)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r68_group % $(16)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r68_group % $(16)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r68_group % $(16)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r68_right = tl.load($(vcat_5_ptr) + (((((((r68_group % $(16)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r68_group % $(16)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r68_group % $(16)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) % $(128)) * $(192)) + (((((r68_group % $(16)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r68_value = tl.dot(r68_left, r68_right)
      r68_value = r68_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r68_group * $(16384)) + (r68_row[:, None] * $(128)) + r68_col[None, :]), r68_value)
    }
    r69_z = tl.arange($(262144))
    r69_value = (tl.load($(vbmm_2_ptr) + ((((r69_z / $(131072)) % $(2)) * $(131072)) + (((r69_z / $(16384)) % $(8)) * $(16384)) + (((r69_z / $(128)) % $(128)) * $(128)) + (r69_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(262144)])
    tl.store($(vmul_25_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(262144))
    r70_value = (tl.load($(vmul_25_ptr) + ((((r70_z / $(131072)) % $(2)) * $(131072)) + (((r70_z / $(16384)) % $(8)) * $(16384)) + (((r70_z / $(128)) % $(128)) * $(128)) + (r70_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r70_z / $(131072)) % $(2)) * $(16384)) + (((r70_z / $(128)) % $(128)) * $(128)) + (r70_z % $(128))), dtype=tl.bfloat16))
    r70_value = r70_value.to(tl.bfloat16)
    r70_value = tl.broadcast_to(r70_value, [$(262144)])
    tl.store($(vadd_6_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(262144))
    r71_value = tl.load($(vadd_6_ptr) + ((((r71_z / $(131072)) % $(2)) * $(131072)) + (((r71_z / $(16384)) % $(8)) * $(16384)) + (((r71_z / $(128)) % $(128)) * $(128)) + (r71_z % $(128))), dtype=tl.bfloat16)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(262144)])
    tl.store($(vto_copy_24_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(262144))
    r72_lane = tl.arange($(128))
    r72_row_values = tl.load($(vto_copy_24_ptr) + ((((r72_z[:, None] / $(131072)) % $(2)) * $(131072)) + (((r72_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r72_z[:, None] / $(128)) % $(128)) * $(128)) + (r72_lane[None, :] % $(128))), dtype=tl.float32)
    r72_maximum = tl.max(r72_row_values, axis=1, keep_dims=true)
    r72_weights = tl.exp(r72_row_values - r72_maximum)
    r72_total = tl.sum(r72_weights, axis=1)
    r72_maximum = tl.max(r72_row_values, axis=1)
    r72_value = tl.exp(tl.load($(vto_copy_24_ptr) + ((((r72_z / $(131072)) % $(2)) * $(131072)) + (((r72_z / $(16384)) % $(8)) * $(16384)) + (((r72_z / $(128)) % $(128)) * $(128)) + (r72_z % $(128))), dtype=tl.float32) - r72_maximum) / r72_total
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(262144)])
    tl.store($(vsoftmax_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(262144))
    r73_value = tl.load($(vsoftmax_ptr) + ((((r73_z / $(131072)) % $(2)) * $(131072)) + (((r73_z / $(16384)) % $(8)) * $(16384)) + (((r73_z / $(128)) % $(128)) * $(128)) + (r73_z % $(128))), dtype=tl.float32)
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(262144)])
    tl.store($(vto_copy_25_ptr) + r73_z, r73_value)
    r74_row = tl.arange($(128))
    r74_col = tl.arange($(128))
    r74_lane = tl.arange($(128))
    for r74_group in range($(16)) {
      r74_left = tl.load($(vto_copy_25_ptr) + (((((((r74_group % $(16)) * $(16384)) + (r74_row[:, None] * $(128)) + r74_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r74_group % $(16)) * $(16384)) + (r74_row[:, None] * $(128)) + r74_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r74_group % $(16)) * $(16384)) + (r74_row[:, None] * $(128)) + r74_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r74_group % $(16)) * $(16384)) + (r74_row[:, None] * $(128)) + r74_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r74_right = tl.load($(vmm_3_ptr) + (((((((r74_group % $(16)) * $(16384)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) / $(131072)) % $(2)) * $(262144)) + ((((((r74_group % $(16)) * $(16384)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) / $(128)) % $(128)) * $(2048)) + ((((((r74_group % $(16)) * $(16384)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) / $(16384)) % $(8)) * $(256)) + ((((r74_group % $(16)) * $(16384)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r74_value = tl.dot(r74_left, r74_right)
      r74_value = r74_value.to(tl.bfloat16)
      tl.store($(vbmm_3_ptr) + ((r74_group * $(16384)) + (r74_row[:, None] * $(128)) + r74_col[None, :]), r74_value)
    }
    r75_row = tl.arange($(256))
    r75_col = tl.arange($(512))
    r75_lane = tl.arange($(1024))
    r75_left = tl.load($(vbmm_3_ptr) + ((((((r75_row[:, None] * $(1024)) + r75_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + (((((((((r75_row[:, None] * $(1024)) + r75_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + (((((r75_row[:, None] * $(1024)) + r75_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r75_row[:, None] * $(1024)) + r75_lane[None, :]) % $(1024))) / $(128)) % $(8)) * $(16384)) + (((((r75_row[:, None] * $(1024)) + r75_lane[None, :]) / $(1024)) % $(128)) * $(128)) + (((((((r75_row[:, None] * $(1024)) + r75_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + (((((r75_row[:, None] * $(1024)) + r75_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r75_row[:, None] * $(1024)) + r75_lane[None, :]) % $(1024))) % $(128))), dtype=tl.bfloat16)
    r75_right = tl.load($(vto_copy_6_ptr) + (((r75_col[None, :] % $(512)) * $(1024)) + (r75_lane[:, None] % $(1024))), dtype=tl.bfloat16)
    r75_value = tl.dot(r75_left, r75_right)
    r75_value = r75_value.to(tl.bfloat16)
    tl.store($(vmm_4_ptr) + ((r75_row[:, None] * $(512)) + r75_col[None, :]), r75_value)
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
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_ptr : RegionName := "vsub"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let vmm_4_ptr : RegionName := "vmm_4"
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
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vpow_4_ptr : RegionName := "candidate_vpow_4"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vpow_5_ptr : RegionName := "candidate_vpow_5"
  let candidate_vreciprocal_2_ptr : RegionName := "candidate_vreciprocal_2"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vcos_2_ptr : RegionName := "candidate_vcos_2"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vsin_2_ptr : RegionName := "candidate_vsin_2"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vcos_3_ptr : RegionName := "candidate_vcos_3"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vsin_3_ptr : RegionName := "candidate_vsin_3"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vstack_1_ptr : RegionName := "candidate_vstack_1"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vcat_2_ptr : RegionName := "candidate_vcat_2"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
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
    tl.store($(candidate_vto_copy_20_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(65536))
    r18_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r18_z / $(32768)) % $(2)) * $(32768)) + (((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_20_ptr) + ((((r18_z / $(32768)) % $(2)) * $(32768)) + (((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(65536)])
    tl.store($(candidate_vpow_3_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(256))
    r19_lane = tl.arange($(256))
    r19_value = (tl.sum(tl.load($(candidate_vpow_3_ptr) + ((((r19_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r19_z[:, None] % $(128)) * $(256)) + r19_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
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
    r22_value = (tl.load($(candidate_vto_copy_20_ptr) + ((((r22_z / $(32768)) % $(2)) * $(32768)) + (((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r22_z / $(32768)) % $(2)) * $(128)) + ((r22_z / $(256)) % $(128))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(65536)])
    tl.store($(candidate_vmul_13_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(65536))
    r23_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r23_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_13_ptr) + ((((r23_z / $(32768)) % $(2)) * $(32768)) + (((r23_z / $(256)) % $(128)) * $(256)) + (r23_z % $(256))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(65536)])
    tl.store($(candidate_vmul_14_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = tl.load($(candidate_vmul_14_ptr) + ((((r24_z / $(32768)) % $(2)) * $(32768)) + (((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.float32)
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(candidate_vto_copy_21_ptr) + r24_z, r24_value)
    r25_row = tl.arange($(256))
    r25_col = tl.arange($(1536))
    r25_lane = tl.arange($(256))
    r25_left = tl.load($(candidate_vto_copy_21_ptr) + ((((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r25_row[:, None] * $(256)) + r25_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r25_right = tl.load($(candidate_vto_copy_2_ptr) + (((r25_col[None, :] % $(1536)) * $(256)) + (r25_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r25_value = tl.dot(r25_left, r25_right)
    r25_value = r25_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r25_row[:, None] * $(1536)) + r25_col[None, :]), r25_value)
    r26_row = tl.arange($(256))
    r26_col = tl.arange($(576))
    r26_lane = tl.arange($(512))
    r26_left = tl.load($(input0_ptr) + ((((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r26_row[:, None] * $(512)) + r26_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r26_right = tl.load($(candidate_vto_copy_3_ptr) + (((r26_col[None, :] % $(576)) * $(512)) + (r26_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r26_value = tl.dot(r26_left, r26_right)
    r26_value = r26_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r26_row[:, None] * $(576)) + r26_col[None, :]), r26_value)
    r27_z = tl.arange($(131072))
    r27_value = tl.load($(candidate_vmm_2_ptr) + ((((r27_z / $(65536)) % $(2)) * $(73728)) + (((r27_z / $(512)) % $(128)) * $(576)) + (r27_z % $(512))), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(131072)])
    tl.store($(candidate_vto_copy_22_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(131072))
    r28_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r28_z / $(65536)) % $(2)) * $(65536)) + (((r28_z / $(512)) % $(128)) * $(512)) + (r28_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_22_ptr) + ((((r28_z / $(65536)) % $(2)) * $(65536)) + (((r28_z / $(512)) % $(128)) * $(512)) + (r28_z % $(512))), dtype=tl.float32)))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(131072)])
    tl.store($(candidate_vpow_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(256))
    r29_lane = tl.arange($(512))
    r29_value = (tl.sum(tl.load($(candidate_vpow_4_ptr) + ((((r29_z[:, None] / $(128)) % $(2)) * $(65536)) + ((r29_z[:, None] % $(128)) * $(512)) + r29_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
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
    r32_z = tl.arange($(131072))
    r32_value = (tl.load($(candidate_vto_copy_22_ptr) + ((((r32_z / $(65536)) % $(2)) * $(65536)) + (((r32_z / $(512)) % $(128)) * $(512)) + (r32_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((((r32_z / $(65536)) % $(2)) * $(128)) + ((r32_z / $(512)) % $(128))), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(131072)])
    tl.store($(candidate_vmul_15_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(131072))
    r33_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r33_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_15_ptr) + ((((r33_z / $(65536)) % $(2)) * $(65536)) + (((r33_z / $(512)) % $(128)) * $(512)) + (r33_z % $(512))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(131072)])
    tl.store($(candidate_vmul_16_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(131072))
    r34_value = tl.load($(candidate_vmul_16_ptr) + ((((r34_z / $(65536)) % $(2)) * $(65536)) + (((r34_z / $(512)) % $(128)) * $(512)) + (r34_z % $(512))), dtype=tl.float32)
    r34_value = r34_value.to(tl.bfloat16)
    r34_value = tl.broadcast_to(r34_value, [$(131072)])
    tl.store($(candidate_vto_copy_23_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(32))
    r35_value = ((tl.cast(tl.toReal(r35_z), tl.float32) * 2.0) / 64.0)
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(32)])
    tl.store($(candidate_vdiv_2_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(32))
    r36_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(candidate_vdiv_2_ptr) + r36_z, dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(32)])
    tl.store($(candidate_vpow_5_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(32))
    r37_value = 1.0 / tl.load($(candidate_vpow_5_ptr) + r37_z, dtype=tl.float32)
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(32)])
    tl.store($(candidate_vreciprocal_2_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(32))
    r38_value = (tl.load($(candidate_vreciprocal_2_ptr) + r38_z, dtype=tl.float32) * 1.0)
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(32)])
    tl.store($(candidate_vmul_17_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(4096))
    r39_value = (tl.cast(tl.toReal(((r39_z / $(32)) % $(128))), tl.float32) * tl.load($(candidate_vmul_17_ptr) + (r39_z % $(32)), dtype=tl.float32))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(4096)])
    tl.store($(candidate_vmul_18_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(8192))
    r40_value = tl.cos(tl.load($(candidate_vmul_18_ptr) + ((((r40_z / $(64)) % $(128)) * $(32)) + ((((((r40_z / $(64)) % $(128)) * $(64)) + (r40_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(8192)])
    tl.store($(candidate_vcos_2_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(8192))
    r41_value = tl.load($(candidate_vcos_2_ptr) + ((((r41_z / $(64)) % $(128)) * $(64)) + (r41_z % $(64))), dtype=tl.float32)
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(8192)])
    tl.store($(candidate_vto_copy_24_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(8192))
    r42_value = tl.sin(tl.load($(candidate_vmul_18_ptr) + ((((r42_z / $(64)) % $(128)) * $(32)) + ((((((r42_z / $(64)) % $(128)) * $(64)) + (r42_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(8192)])
    tl.store($(candidate_vsin_2_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(8192))
    r43_value = tl.load($(candidate_vsin_2_ptr) + ((((r43_z / $(64)) % $(128)) * $(64)) + (r43_z % $(64))), dtype=tl.float32)
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(8192)])
    tl.store($(candidate_vto_copy_25_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(65536))
    r44_value = -tl.load($(candidate_vmm_1_ptr) + ((((r44_z / $(256)) % $(2)) * $(196608)) + (((r44_z / $(512)) % $(128)) * $(1536)) + (((r44_z / $(32)) % $(8)) * $(192)) + ((r44_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16)
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(65536)])
    tl.store($(candidate_vneg_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(131072))
    r45_value = tl.where((r45_z % $(2)) < $(1), tl.load($(candidate_vneg_ptr) + ((((r45_z / $(1024)) % $(128)) * $(512)) + (((r45_z / $(512)) % $(2)) * $(256)) + (((r45_z / $(64)) % $(8)) * $(32)) + ((r45_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_1_ptr) + ((((r45_z / $(512)) % $(2)) * $(196608)) + (((r45_z / $(1024)) % $(128)) * $(1536)) + (((r45_z / $(64)) % $(8)) * $(192)) + (((r45_z / $(2)) % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(131072)])
    tl.store($(candidate_vstack_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(131072))
    r46_value = (tl.load($(candidate_vmm_1_ptr) + ((((r46_z / $(512)) % $(2)) * $(196608)) + (((r46_z / $(1024)) % $(128)) * $(1536)) + (((r46_z / $(64)) % $(8)) * $(192)) + (r46_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_24_ptr) + ((((r46_z / $(1024)) % $(128)) * $(64)) + (r46_z % $(64))), dtype=tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(131072)])
    tl.store($(candidate_vmul_19_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(131072))
    r47_value = (tl.load($(candidate_vstack_ptr) + ((((r47_z / $(1024)) % $(128)) * $(1024)) + (((r47_z / $(512)) % $(2)) * $(512)) + (((r47_z / $(64)) % $(8)) * $(64)) + (((((((r47_z / $(1024)) % $(128)) * $(1024)) + (((r47_z / $(512)) % $(2)) * $(512)) + (((r47_z / $(64)) % $(8)) * $(64)) + (r47_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r47_z / $(1024)) % $(128)) * $(1024)) + (((r47_z / $(512)) % $(2)) * $(512)) + (((r47_z / $(64)) % $(8)) * $(64)) + (r47_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r47_z / $(1024)) % $(128)) * $(64)) + (r47_z % $(64))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(131072)])
    tl.store($(candidate_vmul_20_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(131072))
    r48_value = (tl.load($(candidate_vmul_19_ptr) + ((((r48_z / $(1024)) % $(128)) * $(1024)) + (((r48_z / $(512)) % $(2)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (r48_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_20_ptr) + ((((r48_z / $(1024)) % $(128)) * $(1024)) + (((r48_z / $(512)) % $(2)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (r48_z % $(64))), dtype=tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(131072)])
    tl.store($(candidate_vadd_4_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(8192))
    r49_value = tl.cos(tl.load($(candidate_vmul_18_ptr) + ((((r49_z / $(64)) % $(128)) * $(32)) + ((((((r49_z / $(64)) % $(128)) * $(64)) + (r49_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(8192)])
    tl.store($(candidate_vcos_3_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(8192))
    r50_value = tl.load($(candidate_vcos_3_ptr) + ((((r50_z / $(64)) % $(128)) * $(64)) + (r50_z % $(64))), dtype=tl.float32)
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(8192)])
    tl.store($(candidate_vto_copy_26_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(8192))
    r51_value = tl.sin(tl.load($(candidate_vmul_18_ptr) + ((((r51_z / $(64)) % $(128)) * $(32)) + ((((((r51_z / $(64)) % $(128)) * $(64)) + (r51_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(8192)])
    tl.store($(candidate_vsin_3_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(8192))
    r52_value = tl.load($(candidate_vsin_3_ptr) + ((((r52_z / $(64)) % $(128)) * $(64)) + (r52_z % $(64))), dtype=tl.float32)
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(8192)])
    tl.store($(candidate_vto_copy_27_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(8192))
    r53_value = -tl.load($(candidate_vmm_2_ptr) + ((((r53_z / $(32)) % $(2)) * $(73728)) + (((r53_z / $(64)) % $(128)) * $(576)) + ((r53_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16)
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(8192)])
    tl.store($(candidate_vneg_1_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(16384))
    r54_value = tl.where((r54_z % $(2)) < $(1), tl.load($(candidate_vneg_1_ptr) + ((((r54_z / $(128)) % $(128)) * $(64)) + (((r54_z / $(64)) % $(2)) * $(32)) + ((r54_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_2_ptr) + ((((r54_z / $(64)) % $(2)) * $(73728)) + (((r54_z / $(128)) % $(128)) * $(576)) + (((r54_z / $(2)) % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(16384)])
    tl.store($(candidate_vstack_1_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(16384))
    r55_value = (tl.load($(candidate_vmm_2_ptr) + ((((r55_z / $(64)) % $(2)) * $(73728)) + (((r55_z / $(128)) % $(128)) * $(576)) + (r55_z % $(64)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r55_z / $(128)) % $(128)) * $(64)) + (r55_z % $(64))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(16384)])
    tl.store($(candidate_vmul_21_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(16384))
    r56_value = (tl.load($(candidate_vstack_1_ptr) + ((((r56_z / $(128)) % $(128)) * $(128)) + (((r56_z / $(64)) % $(2)) * $(64)) + (((((((r56_z / $(128)) % $(128)) * $(128)) + (((r56_z / $(64)) % $(2)) * $(64)) + (r56_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r56_z / $(128)) % $(128)) * $(128)) + (((r56_z / $(64)) % $(2)) * $(64)) + (r56_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r56_z / $(128)) % $(128)) * $(64)) + (r56_z % $(64))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(16384)])
    tl.store($(candidate_vmul_22_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(16384))
    r57_value = (tl.load($(candidate_vmul_21_ptr) + ((((r57_z / $(128)) % $(128)) * $(128)) + (((r57_z / $(64)) % $(2)) * $(64)) + (r57_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_22_ptr) + ((((r57_z / $(128)) % $(128)) * $(128)) + (((r57_z / $(64)) % $(2)) * $(64)) + (r57_z % $(64))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(16384)])
    tl.store($(candidate_vadd_5_ptr) + r57_z, r57_value)
    r58_row = tl.arange($(256))
    r58_col = tl.arange($(512))
    r58_lane = tl.arange($(128))
    for r58_group in range($(8)) {
      r58_left = tl.load($(candidate_vmm_1_ptr) + (((((((((((r58_group % $(8)) * $(32768)) + (r58_row[:, None] * $(128)) + r58_lane[None, :]) / $(128)) % $(256)) * $(1024)) + ((((((r58_group % $(8)) * $(32768)) + (r58_row[:, None] * $(128)) + r58_lane[None, :]) / $(32768)) % $(8)) * $(128)) + ((((r58_group % $(8)) * $(32768)) + (r58_row[:, None] * $(128)) + r58_lane[None, :]) % $(128))) / $(131072)) % $(2)) * $(196608)) + (((((((r58_group % $(8)) * $(32768)) + (r58_row[:, None] * $(128)) + r58_lane[None, :]) / $(128)) % $(256)) % $(128)) * $(1536)) + ((((((r58_group % $(8)) * $(32768)) + (r58_row[:, None] * $(128)) + r58_lane[None, :]) / $(32768)) % $(8)) * $(192)) + ((((r58_group % $(8)) * $(32768)) + (r58_row[:, None] * $(128)) + r58_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r58_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r58_group % $(8)) * $(65536)) + (r58_lane[:, None] * $(512)) + r58_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r58_group % $(8)) * $(65536)) + (r58_lane[:, None] * $(512)) + r58_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r58_group % $(8)) * $(65536)) + (r58_lane[:, None] * $(512)) + r58_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r58_value = tl.dot(r58_left, r58_right)
      r58_value = r58_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r58_group * $(131072)) + (r58_row[:, None] * $(512)) + r58_col[None, :]), r58_value)
    }
    r59_z = tl.arange($(65536))
    r59_value = tl.load($(candidate_vadd_4_ptr) + ((((r59_z / $(64)) % $(128)) * $(1024)) + (((r59_z / $(8192)) % $(8)) * $(64)) + (r59_z % $(64))), dtype=tl.bfloat16)
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(65536)])
    tl.store($(candidate_vto_copy_28_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(8192))
    r60_value = tl.load($(candidate_vadd_5_ptr) + ((((r60_z / $(64)) % $(128)) * $(128)) + (r60_z % $(64))), dtype=tl.bfloat16)
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(8192)])
    tl.store($(candidate_vto_copy_29_ptr) + r60_z, r60_value)
    r61_row = tl.arange($(128))
    r61_col = tl.arange($(128))
    r61_lane = tl.arange($(64))
    for r61_group in range($(8)) {
      r61_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r61_group % $(8)) * $(8192)) + (r61_row[:, None] * $(64)) + r61_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r61_group % $(8)) * $(8192)) + (r61_row[:, None] * $(64)) + r61_lane[None, :]) / $(64)) % $(128)) * $(64)) + ((((r61_group % $(8)) * $(8192)) + (r61_row[:, None] * $(64)) + r61_lane[None, :]) % $(64))), dtype=tl.float32)
      r61_right = tl.load($(candidate_vto_copy_29_ptr) + ((((((r61_group % $(8)) * $(8192)) + (r61_lane[:, None] * $(128)) + r61_col[None, :]) % $(128)) * $(64)) + (((((r61_group % $(8)) * $(8192)) + (r61_lane[:, None] * $(128)) + r61_col[None, :]) / $(128)) % $(64))), dtype=tl.float32)
      r61_value = tl.dot(r61_left, r61_right)
      r61_value = r61_value.to(tl.float32)
      tl.store($(candidate_vbmm_3_ptr) + ((r61_group * $(16384)) + (r61_row[:, None] * $(128)) + r61_col[None, :]), r61_value)
    }
    r62_z = tl.arange($(524288))
    r62_value = tl.load($(candidate_vbmm_2_ptr) + ((((r62_z / $(65536)) % $(8)) * $(131072)) + (((r62_z / $(512)) % $(128)) * $(512)) + (r62_z % $(512))), dtype=tl.bfloat16)
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(524288)])
    tl.store($(candidate_vto_copy_30_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(65536))
    r63_value = tl.load($(candidate_vto_copy_23_ptr) + ((((r63_z / $(512)) % $(128)) * $(512)) + (r63_z % $(512))), dtype=tl.bfloat16)
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(65536)])
    tl.store($(candidate_vto_copy_31_ptr) + r63_z, r63_value)
    r64_row = tl.arange($(1024))
    r64_col = tl.arange($(128))
    r64_lane = tl.arange($(512))
    r64_left = tl.load($(candidate_vto_copy_30_ptr) + ((((((r64_row[:, None] * $(512)) + r64_lane[None, :]) / $(65536)) % $(8)) * $(65536)) + (((((r64_row[:, None] * $(512)) + r64_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r64_row[:, None] * $(512)) + r64_lane[None, :]) % $(512))), dtype=tl.float32)
    r64_right = tl.load($(candidate_vto_copy_31_ptr) + (((((r64_lane[:, None] * $(128)) + r64_col[None, :]) % $(128)) * $(512)) + ((((r64_lane[:, None] * $(128)) + r64_col[None, :]) / $(128)) % $(512))), dtype=tl.float32)
    r64_value = tl.dot(r64_left, r64_right)
    r64_value = r64_value.to(tl.float32)
    tl.store($(candidate_vmm_3_ptr) + ((r64_row[:, None] * $(128)) + r64_col[None, :]), r64_value)
    r65_z = tl.arange($(131072))
    r65_value = (tl.load($(candidate_vbmm_3_ptr) + ((((r65_z / $(16384)) % $(8)) * $(16384)) + (((r65_z / $(128)) % $(128)) * $(128)) + (r65_z % $(128))), dtype=tl.float32) + tl.load($(candidate_vmm_3_ptr) + ((((r65_z / $(16384)) % $(8)) * $(16384)) + (((r65_z / $(128)) % $(128)) * $(128)) + (r65_z % $(128))), dtype=tl.float32))
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(131072)])
    tl.store($(candidate_vadd_6_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(131072))
    r66_value = (tl.load($(candidate_vadd_6_ptr) + ((((r66_z / $(16384)) % $(8)) * $(16384)) + (((r66_z / $(128)) % $(128)) * $(128)) + (r66_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(131072)])
    tl.store($(candidate_vmul_23_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(128))
    r67_value = (tl.cast(r67_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r67_value = tl.broadcast_to(r67_value, [$(128)])
    tl.store($(candidate_vadd_7_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(128))
    r68_value = (tl.load($(candidate_vadd_7_ptr) + r68_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r68_value = tl.broadcast_to(r68_value, [$(128)])
    tl.store($(candidate_vsub_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(16384))
    r69_value = (tl.cast((r69_z % $(128)), tl.int64) > tl.load($(candidate_vsub_ptr) + ((r69_z / $(128)) % $(128)), dtype=tl.int64))
    r69_value = tl.broadcast_to(r69_value, [$(16384)])
    tl.store($(candidate_vgt_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(131072))
    r70_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r70_z / $(128)) % $(128)) * $(128)) + (r70_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r70_z / $(16384)) % $(8)) * $(16384)) + (((r70_z / $(128)) % $(128)) * $(128)) + (r70_z % $(128))), dtype=tl.float32))
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(131072))
    r71_lane = tl.arange($(128))
    r71_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r71_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r71_z[:, None] / $(128)) % $(128)) * $(128)) + (r71_lane[None, :] % $(128))), dtype=tl.float32)
    r71_maximum = tl.max(r71_row_values, axis=1, keep_dims=true)
    r71_weights = tl.exp(r71_row_values - r71_maximum)
    r71_total = tl.sum(r71_weights, axis=1)
    r71_maximum = tl.max(r71_row_values, axis=1)
    r71_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r71_z / $(16384)) % $(8)) * $(16384)) + (((r71_z / $(128)) % $(128)) * $(128)) + (r71_z % $(128))), dtype=tl.float32) - r71_maximum) / r71_total
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(131072)])
    tl.store($(candidate_vsoftmax_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(131072))
    r72_value = tl.load($(candidate_vsoftmax_ptr) + ((((r72_z / $(16384)) % $(8)) * $(16384)) + (((r72_z / $(128)) % $(128)) * $(128)) + (r72_z % $(128))), dtype=tl.float32)
    r72_value = r72_value.to(tl.bfloat16)
    r72_value = tl.broadcast_to(r72_value, [$(131072)])
    tl.store($(candidate_vto_copy_32_ptr) + r72_z, r72_value)
    r73_row = tl.arange($(1024))
    r73_col = tl.arange($(512))
    r73_lane = tl.arange($(128))
    r73_left = tl.load($(candidate_vto_copy_32_ptr) + ((((((r73_row[:, None] * $(128)) + r73_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r73_row[:, None] * $(128)) + r73_lane[None, :]) / $(128)) % $(128)) * $(128)) + (((r73_row[:, None] * $(128)) + r73_lane[None, :]) % $(128))), dtype=tl.bfloat16)
    r73_right = tl.load($(candidate_vto_copy_23_ptr) + ((((((r73_lane[:, None] * $(512)) + r73_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r73_lane[:, None] * $(512)) + r73_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r73_value = tl.dot(r73_left, r73_right)
    r73_value = r73_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r73_row[:, None] * $(512)) + r73_col[None, :]), r73_value)
    r74_z = tl.arange($(65536))
    r74_value = tl.load($(candidate_vadd_4_ptr) + ((((r74_z / $(64)) % $(128)) * $(1024)) + (((r74_z / $(8192)) % $(8)) * $(64)) + (r74_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(65536)])
    tl.store($(candidate_vto_copy_33_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(8192))
    r75_value = tl.load($(candidate_vadd_5_ptr) + ((((r75_z / $(64)) % $(128)) * $(128)) + (r75_z % $(64)) + $(64)), dtype=tl.bfloat16)
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(8192)])
    tl.store($(candidate_vto_copy_34_ptr) + r75_z, r75_value)
    r76_row = tl.arange($(128))
    r76_col = tl.arange($(128))
    r76_lane = tl.arange($(64))
    for r76_group in range($(8)) {
      r76_left = tl.load($(candidate_vto_copy_33_ptr) + (((((((r76_group % $(8)) * $(8192)) + (r76_row[:, None] * $(64)) + r76_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r76_group % $(8)) * $(8192)) + (r76_row[:, None] * $(64)) + r76_lane[None, :]) / $(64)) % $(128)) * $(64)) + ((((r76_group % $(8)) * $(8192)) + (r76_row[:, None] * $(64)) + r76_lane[None, :]) % $(64))), dtype=tl.float32)
      r76_right = tl.load($(candidate_vto_copy_34_ptr) + ((((((r76_group % $(8)) * $(8192)) + (r76_lane[:, None] * $(128)) + r76_col[None, :]) % $(128)) * $(64)) + (((((r76_group % $(8)) * $(8192)) + (r76_lane[:, None] * $(128)) + r76_col[None, :]) / $(128)) % $(64))), dtype=tl.float32)
      r76_value = tl.dot(r76_left, r76_right)
      r76_value = r76_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r76_group * $(16384)) + (r76_row[:, None] * $(128)) + r76_col[None, :]), r76_value)
    }
    r77_z = tl.arange($(524288))
    r77_value = tl.load($(candidate_vbmm_2_ptr) + ((((r77_z / $(65536)) % $(8)) * $(131072)) + (((r77_z / $(512)) % $(128)) * $(512)) + (r77_z % $(512)) + $(65536)), dtype=tl.bfloat16)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(524288)])
    tl.store($(candidate_vto_copy_35_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(65536))
    r78_value = tl.load($(candidate_vto_copy_23_ptr) + ((((r78_z / $(512)) % $(128)) * $(512)) + (r78_z % $(512)) + $(65536)), dtype=tl.bfloat16)
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(65536)])
    tl.store($(candidate_vto_copy_36_ptr) + r78_z, r78_value)
    r79_row = tl.arange($(1024))
    r79_col = tl.arange($(128))
    r79_lane = tl.arange($(512))
    r79_left = tl.load($(candidate_vto_copy_35_ptr) + ((((((r79_row[:, None] * $(512)) + r79_lane[None, :]) / $(65536)) % $(8)) * $(65536)) + (((((r79_row[:, None] * $(512)) + r79_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r79_row[:, None] * $(512)) + r79_lane[None, :]) % $(512))), dtype=tl.float32)
    r79_right = tl.load($(candidate_vto_copy_36_ptr) + (((((r79_lane[:, None] * $(128)) + r79_col[None, :]) % $(128)) * $(512)) + ((((r79_lane[:, None] * $(128)) + r79_col[None, :]) / $(128)) % $(512))), dtype=tl.float32)
    r79_value = tl.dot(r79_left, r79_right)
    r79_value = r79_value.to(tl.float32)
    tl.store($(candidate_vmm_5_ptr) + ((r79_row[:, None] * $(128)) + r79_col[None, :]), r79_value)
    r80_z = tl.arange($(131072))
    r80_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r80_z / $(16384)) % $(8)) * $(16384)) + (((r80_z / $(128)) % $(128)) * $(128)) + (r80_z % $(128))), dtype=tl.float32) + tl.load($(candidate_vmm_5_ptr) + ((((r80_z / $(16384)) % $(8)) * $(16384)) + (((r80_z / $(128)) % $(128)) * $(128)) + (r80_z % $(128))), dtype=tl.float32))
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(131072)])
    tl.store($(candidate_vadd_8_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(131072))
    r81_value = (tl.load($(candidate_vadd_8_ptr) + ((((r81_z / $(16384)) % $(8)) * $(16384)) + (((r81_z / $(128)) % $(128)) * $(128)) + (r81_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(131072)])
    tl.store($(candidate_vmul_24_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(128))
    r82_value = (tl.cast(r82_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r82_value = tl.broadcast_to(r82_value, [$(128)])
    tl.store($(candidate_vadd_9_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(128))
    r83_value = (tl.load($(candidate_vadd_9_ptr) + r83_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r83_value = tl.broadcast_to(r83_value, [$(128)])
    tl.store($(candidate_vsub_1_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(16384))
    r84_value = (tl.cast((r84_z % $(128)), tl.int64) > tl.load($(candidate_vsub_1_ptr) + ((r84_z / $(128)) % $(128)), dtype=tl.int64))
    r84_value = tl.broadcast_to(r84_value, [$(16384)])
    tl.store($(candidate_vgt_1_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(131072))
    r85_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r85_z / $(128)) % $(128)) * $(128)) + (r85_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r85_z / $(16384)) % $(8)) * $(16384)) + (((r85_z / $(128)) % $(128)) * $(128)) + (r85_z % $(128))), dtype=tl.float32))
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(131072))
    r86_lane = tl.arange($(128))
    r86_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r86_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r86_z[:, None] / $(128)) % $(128)) * $(128)) + (r86_lane[None, :] % $(128))), dtype=tl.float32)
    r86_maximum = tl.max(r86_row_values, axis=1, keep_dims=true)
    r86_weights = tl.exp(r86_row_values - r86_maximum)
    r86_total = tl.sum(r86_weights, axis=1)
    r86_maximum = tl.max(r86_row_values, axis=1)
    r86_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(128)) % $(128)) * $(128)) + (r86_z % $(128))), dtype=tl.float32) - r86_maximum) / r86_total
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(131072)])
    tl.store($(candidate_vsoftmax_1_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(131072))
    r87_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r87_z / $(16384)) % $(8)) * $(16384)) + (((r87_z / $(128)) % $(128)) * $(128)) + (r87_z % $(128))), dtype=tl.float32)
    r87_value = r87_value.to(tl.bfloat16)
    r87_value = tl.broadcast_to(r87_value, [$(131072)])
    tl.store($(candidate_vto_copy_37_ptr) + r87_z, r87_value)
    r88_row = tl.arange($(1024))
    r88_col = tl.arange($(512))
    r88_lane = tl.arange($(128))
    r88_left = tl.load($(candidate_vto_copy_37_ptr) + ((((((r88_row[:, None] * $(128)) + r88_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r88_row[:, None] * $(128)) + r88_lane[None, :]) / $(128)) % $(128)) * $(128)) + (((r88_row[:, None] * $(128)) + r88_lane[None, :]) % $(128))), dtype=tl.bfloat16)
    r88_right = tl.load($(candidate_vto_copy_23_ptr) + ((((((r88_lane[:, None] * $(512)) + r88_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r88_lane[:, None] * $(512)) + r88_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
    r88_value = tl.dot(r88_left, r88_right)
    r88_value = r88_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r88_row[:, None] * $(512)) + r88_col[None, :]), r88_value)
    r89_z = tl.arange($(1048576))
    r89_value = tl.where(((r89_z / $(4096)) % $(256)) < $(128), tl.load($(candidate_vmm_4_ptr) + (((((((r89_z / $(512)) % $(8)) * $(128)) + (((r89_z / $(4096)) % $(256)) - $(0))) % $(1024)) * $(512)) + (r89_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_6_ptr) + (((((((r89_z / $(512)) % $(8)) * $(128)) + (((r89_z / $(4096)) % $(256)) - $(128))) % $(1024)) * $(512)) + (r89_z % $(512))), dtype=tl.bfloat16))
    r89_value = r89_value.to(tl.bfloat16)
    r89_value = tl.broadcast_to(r89_value, [$(1048576)])
    tl.store($(candidate_vcat_2_ptr) + r89_z, r89_value)
    r90_row = tl.arange($(256))
    r90_col = tl.arange($(128))
    r90_lane = tl.arange($(512))
    for r90_group in range($(8)) {
      r90_left = tl.load($(candidate_vcat_2_ptr) + (((((((r90_group % $(8)) * $(131072)) + (r90_row[:, None] * $(512)) + r90_lane[None, :]) / $(512)) % $(256)) * $(4096)) + ((((((r90_group % $(8)) * $(131072)) + (r90_row[:, None] * $(512)) + r90_lane[None, :]) / $(131072)) % $(8)) * $(512)) + ((((r90_group % $(8)) * $(131072)) + (r90_row[:, None] * $(512)) + r90_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r90_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r90_group % $(8)) * $(65536)) + (r90_lane[:, None] * $(128)) + r90_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r90_group % $(8)) * $(65536)) + (r90_lane[:, None] * $(128)) + r90_col[None, :]) % $(128)) * $(512)) + (((((r90_group % $(8)) * $(65536)) + (r90_lane[:, None] * $(128)) + r90_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r90_value = tl.dot(r90_left, r90_right)
      r90_value = r90_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r90_group * $(32768)) + (r90_row[:, None] * $(128)) + r90_col[None, :]), r90_value)
    }
    r91_row = tl.arange($(256))
    r91_col = tl.arange($(512))
    r91_lane = tl.arange($(1024))
    r91_left = tl.load($(candidate_vbmm_5_ptr) + ((((((((((r91_row[:, None] * $(1024)) + r91_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + (((((r91_row[:, None] * $(1024)) + r91_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r91_row[:, None] * $(1024)) + r91_lane[None, :]) % $(1024))) / $(128)) % $(8)) * $(32768)) + (((((r91_row[:, None] * $(1024)) + r91_lane[None, :]) / $(131072)) % $(2)) * $(16384)) + (((((r91_row[:, None] * $(1024)) + r91_lane[None, :]) / $(1024)) % $(128)) * $(128)) + (((((((r91_row[:, None] * $(1024)) + r91_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + (((((r91_row[:, None] * $(1024)) + r91_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r91_row[:, None] * $(1024)) + r91_lane[None, :]) % $(1024))) % $(128))), dtype=tl.bfloat16)
    r91_right = tl.load($(candidate_vto_copy_6_ptr) + (((r91_col[None, :] % $(512)) * $(1024)) + (r91_lane[:, None] % $(1024))), dtype=tl.bfloat16)
    r91_value = tl.dot(r91_left, r91_right)
    r91_value = r91_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_7_ptr) + ((r91_row[:, None] * $(512)) + r91_col[None, :]), r91_value)
    z = tl.arange($(131072))
    value = tl.load($(candidate_vmm_7_ptr) + ((((z / $(65536)) % $(2)) * $(65536)) + (((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_absorb_fwd
