import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_grad_norm_foreach_vs_loop
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vlinalg_vector_norm_ptr : RegionName := "vlinalg_vector_norm"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vlinalg_vector_norm_1_ptr : RegionName := "vlinalg_vector_norm_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vlinalg_vector_norm_2_ptr : RegionName := "vlinalg_vector_norm_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vlinalg_vector_norm_3_ptr : RegionName := "vlinalg_vector_norm_3"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vlinalg_vector_norm_4_ptr : RegionName := "vlinalg_vector_norm_4"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vlinalg_vector_norm_5_ptr : RegionName := "vlinalg_vector_norm_5"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vlinalg_vector_norm_6_ptr : RegionName := "vlinalg_vector_norm_6"
  let vto_copy_7_ptr : RegionName := "vto_copy_7"
  let vlinalg_vector_norm_7_ptr : RegionName := "vlinalg_vector_norm_7"
  let vto_copy_8_ptr : RegionName := "vto_copy_8"
  let vlinalg_vector_norm_8_ptr : RegionName := "vlinalg_vector_norm_8"
  let vto_copy_9_ptr : RegionName := "vto_copy_9"
  let vlinalg_vector_norm_9_ptr : RegionName := "vlinalg_vector_norm_9"
  let vto_copy_10_ptr : RegionName := "vto_copy_10"
  let vlinalg_vector_norm_10_ptr : RegionName := "vlinalg_vector_norm_10"
  let vto_copy_11_ptr : RegionName := "vto_copy_11"
  let vlinalg_vector_norm_11_ptr : RegionName := "vlinalg_vector_norm_11"
  let vto_copy_12_ptr : RegionName := "vto_copy_12"
  let vlinalg_vector_norm_12_ptr : RegionName := "vlinalg_vector_norm_12"
  let vto_copy_13_ptr : RegionName := "vto_copy_13"
  let vlinalg_vector_norm_13_ptr : RegionName := "vlinalg_vector_norm_13"
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vlinalg_vector_norm_14_ptr : RegionName := "vlinalg_vector_norm_14"
  let vto_copy_15_ptr : RegionName := "vto_copy_15"
  let vlinalg_vector_norm_15_ptr : RegionName := "vlinalg_vector_norm_15"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vlinalg_vector_norm_16_ptr : RegionName := "vlinalg_vector_norm_16"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vlinalg_vector_norm_17_ptr : RegionName := "vlinalg_vector_norm_17"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vlinalg_vector_norm_18_ptr : RegionName := "vlinalg_vector_norm_18"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vlinalg_vector_norm_19_ptr : RegionName := "vlinalg_vector_norm_19"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vlinalg_vector_norm_20_ptr : RegionName := "vlinalg_vector_norm_20"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vlinalg_vector_norm_21_ptr : RegionName := "vlinalg_vector_norm_21"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vlinalg_vector_norm_22_ptr : RegionName := "vlinalg_vector_norm_22"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vlinalg_vector_norm_23_ptr : RegionName := "vlinalg_vector_norm_23"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vlinalg_vector_norm_24_ptr : RegionName := "vlinalg_vector_norm_24"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vlinalg_vector_norm_25_ptr : RegionName := "vlinalg_vector_norm_25"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vlinalg_vector_norm_26_ptr : RegionName := "vlinalg_vector_norm_26"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vlinalg_vector_norm_27_ptr : RegionName := "vlinalg_vector_norm_27"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vlinalg_vector_norm_28_ptr : RegionName := "vlinalg_vector_norm_28"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vlinalg_vector_norm_29_ptr : RegionName := "vlinalg_vector_norm_29"
  let vto_copy_30_ptr : RegionName := "vto_copy_30"
  let vlinalg_vector_norm_30_ptr : RegionName := "vlinalg_vector_norm_30"
  let vto_copy_31_ptr : RegionName := "vto_copy_31"
  let vlinalg_vector_norm_31_ptr : RegionName := "vlinalg_vector_norm_31"
  let vstack_ptr : RegionName := "vstack"
  let vto_copy_32_ptr : RegionName := "vto_copy_32"
  triton {
    r0_z = tl.arange($(512))
    r0_value = tl.load($(input0_ptr) + r0_z, dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(512)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1))
    r1_lane = tl.arange($(512))
    r1_value = tl.sqrt(tl.sum(tl.load($(vto_copy_ptr) + r1_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_ptr) + r1_lane[None, :], dtype=tl.float32), axis=1))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(1)])
    tl.store($(vlinalg_vector_norm_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = tl.load($(input0_ptr) + (r2_z + $(512)), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(vto_copy_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1))
    r3_lane = tl.arange($(512))
    r3_value = tl.sqrt(tl.sum(tl.load($(vto_copy_1_ptr) + r3_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_1_ptr) + r3_lane[None, :], dtype=tl.float32), axis=1))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(1)])
    tl.store($(vlinalg_vector_norm_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(512))
    r4_value = tl.load($(input0_ptr) + (r4_z + $(1024)), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(512)])
    tl.store($(vto_copy_2_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(1))
    r5_lane = tl.arange($(512))
    r5_value = tl.sqrt(tl.sum(tl.load($(vto_copy_2_ptr) + r5_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_2_ptr) + r5_lane[None, :], dtype=tl.float32), axis=1))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(1)])
    tl.store($(vlinalg_vector_norm_2_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(512))
    r6_value = tl.load($(input0_ptr) + (r6_z + $(1536)), dtype=tl.bfloat16)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(512)])
    tl.store($(vto_copy_3_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1))
    r7_lane = tl.arange($(512))
    r7_value = tl.sqrt(tl.sum(tl.load($(vto_copy_3_ptr) + r7_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_3_ptr) + r7_lane[None, :], dtype=tl.float32), axis=1))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1)])
    tl.store($(vlinalg_vector_norm_3_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = tl.load($(input0_ptr) + (r8_z + $(2048)), dtype=tl.bfloat16)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(vto_copy_4_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(1))
    r9_lane = tl.arange($(512))
    r9_value = tl.sqrt(tl.sum(tl.load($(vto_copy_4_ptr) + r9_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_4_ptr) + r9_lane[None, :], dtype=tl.float32), axis=1))
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(1)])
    tl.store($(vlinalg_vector_norm_4_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_value = tl.load($(input0_ptr) + (r10_z + $(2560)), dtype=tl.bfloat16)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(vto_copy_5_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1))
    r11_lane = tl.arange($(512))
    r11_value = tl.sqrt(tl.sum(tl.load($(vto_copy_5_ptr) + r11_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_5_ptr) + r11_lane[None, :], dtype=tl.float32), axis=1))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(1)])
    tl.store($(vlinalg_vector_norm_5_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_value = tl.load($(input0_ptr) + (r12_z + $(3072)), dtype=tl.bfloat16)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(vto_copy_6_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(1))
    r13_lane = tl.arange($(512))
    r13_value = tl.sqrt(tl.sum(tl.load($(vto_copy_6_ptr) + r13_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_6_ptr) + r13_lane[None, :], dtype=tl.float32), axis=1))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(1)])
    tl.store($(vlinalg_vector_norm_6_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(512))
    r14_value = tl.load($(input0_ptr) + (r14_z + $(3584)), dtype=tl.bfloat16)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(512)])
    tl.store($(vto_copy_7_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(1))
    r15_lane = tl.arange($(512))
    r15_value = tl.sqrt(tl.sum(tl.load($(vto_copy_7_ptr) + r15_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_7_ptr) + r15_lane[None, :], dtype=tl.float32), axis=1))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(1)])
    tl.store($(vlinalg_vector_norm_7_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(512))
    r16_value = tl.load($(input0_ptr) + (r16_z + $(4096)), dtype=tl.bfloat16)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(512)])
    tl.store($(vto_copy_8_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(1))
    r17_lane = tl.arange($(512))
    r17_value = tl.sqrt(tl.sum(tl.load($(vto_copy_8_ptr) + r17_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_8_ptr) + r17_lane[None, :], dtype=tl.float32), axis=1))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(1)])
    tl.store($(vlinalg_vector_norm_8_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(512))
    r18_value = tl.load($(input0_ptr) + (r18_z + $(4608)), dtype=tl.bfloat16)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(512)])
    tl.store($(vto_copy_9_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(1))
    r19_lane = tl.arange($(512))
    r19_value = tl.sqrt(tl.sum(tl.load($(vto_copy_9_ptr) + r19_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_9_ptr) + r19_lane[None, :], dtype=tl.float32), axis=1))
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(1)])
    tl.store($(vlinalg_vector_norm_9_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(512))
    r20_value = tl.load($(input0_ptr) + (r20_z + $(5120)), dtype=tl.bfloat16)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(512)])
    tl.store($(vto_copy_10_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(1))
    r21_lane = tl.arange($(512))
    r21_value = tl.sqrt(tl.sum(tl.load($(vto_copy_10_ptr) + r21_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_10_ptr) + r21_lane[None, :], dtype=tl.float32), axis=1))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(1)])
    tl.store($(vlinalg_vector_norm_10_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(512))
    r22_value = tl.load($(input0_ptr) + (r22_z + $(5632)), dtype=tl.bfloat16)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(512)])
    tl.store($(vto_copy_11_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(1))
    r23_lane = tl.arange($(512))
    r23_value = tl.sqrt(tl.sum(tl.load($(vto_copy_11_ptr) + r23_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_11_ptr) + r23_lane[None, :], dtype=tl.float32), axis=1))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(1)])
    tl.store($(vlinalg_vector_norm_11_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(512))
    r24_value = tl.load($(input0_ptr) + (r24_z + $(6144)), dtype=tl.bfloat16)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(512)])
    tl.store($(vto_copy_12_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(1))
    r25_lane = tl.arange($(512))
    r25_value = tl.sqrt(tl.sum(tl.load($(vto_copy_12_ptr) + r25_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_12_ptr) + r25_lane[None, :], dtype=tl.float32), axis=1))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(1)])
    tl.store($(vlinalg_vector_norm_12_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(512))
    r26_value = tl.load($(input0_ptr) + (r26_z + $(6656)), dtype=tl.bfloat16)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(512)])
    tl.store($(vto_copy_13_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(1))
    r27_lane = tl.arange($(512))
    r27_value = tl.sqrt(tl.sum(tl.load($(vto_copy_13_ptr) + r27_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_13_ptr) + r27_lane[None, :], dtype=tl.float32), axis=1))
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(1)])
    tl.store($(vlinalg_vector_norm_13_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(512))
    r28_value = tl.load($(input0_ptr) + (r28_z + $(7168)), dtype=tl.bfloat16)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(512)])
    tl.store($(vto_copy_14_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(1))
    r29_lane = tl.arange($(512))
    r29_value = tl.sqrt(tl.sum(tl.load($(vto_copy_14_ptr) + r29_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_14_ptr) + r29_lane[None, :], dtype=tl.float32), axis=1))
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(1)])
    tl.store($(vlinalg_vector_norm_14_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(512))
    r30_value = tl.load($(input0_ptr) + (r30_z + $(7680)), dtype=tl.bfloat16)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(512)])
    tl.store($(vto_copy_15_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(1))
    r31_lane = tl.arange($(512))
    r31_value = tl.sqrt(tl.sum(tl.load($(vto_copy_15_ptr) + r31_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_15_ptr) + r31_lane[None, :], dtype=tl.float32), axis=1))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(1)])
    tl.store($(vlinalg_vector_norm_15_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(512))
    r32_value = tl.load($(input0_ptr) + (r32_z + $(8192)), dtype=tl.bfloat16)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(512)])
    tl.store($(vto_copy_16_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(1))
    r33_lane = tl.arange($(512))
    r33_value = tl.sqrt(tl.sum(tl.load($(vto_copy_16_ptr) + r33_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_16_ptr) + r33_lane[None, :], dtype=tl.float32), axis=1))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(1)])
    tl.store($(vlinalg_vector_norm_16_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(512))
    r34_value = tl.load($(input0_ptr) + (r34_z + $(8704)), dtype=tl.bfloat16)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(512)])
    tl.store($(vto_copy_17_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(1))
    r35_lane = tl.arange($(512))
    r35_value = tl.sqrt(tl.sum(tl.load($(vto_copy_17_ptr) + r35_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_17_ptr) + r35_lane[None, :], dtype=tl.float32), axis=1))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(1)])
    tl.store($(vlinalg_vector_norm_17_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(512))
    r36_value = tl.load($(input0_ptr) + (r36_z + $(9216)), dtype=tl.bfloat16)
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(512)])
    tl.store($(vto_copy_18_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(1))
    r37_lane = tl.arange($(512))
    r37_value = tl.sqrt(tl.sum(tl.load($(vto_copy_18_ptr) + r37_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_18_ptr) + r37_lane[None, :], dtype=tl.float32), axis=1))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(1)])
    tl.store($(vlinalg_vector_norm_18_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(512))
    r38_value = tl.load($(input0_ptr) + (r38_z + $(9728)), dtype=tl.bfloat16)
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(512)])
    tl.store($(vto_copy_19_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(1))
    r39_lane = tl.arange($(512))
    r39_value = tl.sqrt(tl.sum(tl.load($(vto_copy_19_ptr) + r39_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_19_ptr) + r39_lane[None, :], dtype=tl.float32), axis=1))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(1)])
    tl.store($(vlinalg_vector_norm_19_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(512))
    r40_value = tl.load($(input0_ptr) + (r40_z + $(10240)), dtype=tl.bfloat16)
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(512)])
    tl.store($(vto_copy_20_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(1))
    r41_lane = tl.arange($(512))
    r41_value = tl.sqrt(tl.sum(tl.load($(vto_copy_20_ptr) + r41_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_20_ptr) + r41_lane[None, :], dtype=tl.float32), axis=1))
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(1)])
    tl.store($(vlinalg_vector_norm_20_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(512))
    r42_value = tl.load($(input0_ptr) + (r42_z + $(10752)), dtype=tl.bfloat16)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(512)])
    tl.store($(vto_copy_21_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(1))
    r43_lane = tl.arange($(512))
    r43_value = tl.sqrt(tl.sum(tl.load($(vto_copy_21_ptr) + r43_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_21_ptr) + r43_lane[None, :], dtype=tl.float32), axis=1))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(1)])
    tl.store($(vlinalg_vector_norm_21_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(512))
    r44_value = tl.load($(input0_ptr) + (r44_z + $(11264)), dtype=tl.bfloat16)
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(512)])
    tl.store($(vto_copy_22_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(1))
    r45_lane = tl.arange($(512))
    r45_value = tl.sqrt(tl.sum(tl.load($(vto_copy_22_ptr) + r45_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_22_ptr) + r45_lane[None, :], dtype=tl.float32), axis=1))
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(1)])
    tl.store($(vlinalg_vector_norm_22_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(512))
    r46_value = tl.load($(input0_ptr) + (r46_z + $(11776)), dtype=tl.bfloat16)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(512)])
    tl.store($(vto_copy_23_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(1))
    r47_lane = tl.arange($(512))
    r47_value = tl.sqrt(tl.sum(tl.load($(vto_copy_23_ptr) + r47_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_23_ptr) + r47_lane[None, :], dtype=tl.float32), axis=1))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(1)])
    tl.store($(vlinalg_vector_norm_23_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(512))
    r48_value = tl.load($(input0_ptr) + (r48_z + $(12288)), dtype=tl.bfloat16)
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(512)])
    tl.store($(vto_copy_24_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(1))
    r49_lane = tl.arange($(512))
    r49_value = tl.sqrt(tl.sum(tl.load($(vto_copy_24_ptr) + r49_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_24_ptr) + r49_lane[None, :], dtype=tl.float32), axis=1))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(1)])
    tl.store($(vlinalg_vector_norm_24_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(512))
    r50_value = tl.load($(input0_ptr) + (r50_z + $(12800)), dtype=tl.bfloat16)
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(512)])
    tl.store($(vto_copy_25_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(1))
    r51_lane = tl.arange($(512))
    r51_value = tl.sqrt(tl.sum(tl.load($(vto_copy_25_ptr) + r51_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_25_ptr) + r51_lane[None, :], dtype=tl.float32), axis=1))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(1)])
    tl.store($(vlinalg_vector_norm_25_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(512))
    r52_value = tl.load($(input0_ptr) + (r52_z + $(13312)), dtype=tl.bfloat16)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(512)])
    tl.store($(vto_copy_26_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(1))
    r53_lane = tl.arange($(512))
    r53_value = tl.sqrt(tl.sum(tl.load($(vto_copy_26_ptr) + r53_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_26_ptr) + r53_lane[None, :], dtype=tl.float32), axis=1))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(1)])
    tl.store($(vlinalg_vector_norm_26_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(512))
    r54_value = tl.load($(input0_ptr) + (r54_z + $(13824)), dtype=tl.bfloat16)
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(512)])
    tl.store($(vto_copy_27_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(1))
    r55_lane = tl.arange($(512))
    r55_value = tl.sqrt(tl.sum(tl.load($(vto_copy_27_ptr) + r55_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_27_ptr) + r55_lane[None, :], dtype=tl.float32), axis=1))
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(1)])
    tl.store($(vlinalg_vector_norm_27_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(512))
    r56_value = tl.load($(input0_ptr) + (r56_z + $(14336)), dtype=tl.bfloat16)
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(512)])
    tl.store($(vto_copy_28_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(1))
    r57_lane = tl.arange($(512))
    r57_value = tl.sqrt(tl.sum(tl.load($(vto_copy_28_ptr) + r57_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_28_ptr) + r57_lane[None, :], dtype=tl.float32), axis=1))
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(1)])
    tl.store($(vlinalg_vector_norm_28_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(512))
    r58_value = tl.load($(input0_ptr) + (r58_z + $(14848)), dtype=tl.bfloat16)
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(512)])
    tl.store($(vto_copy_29_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(1))
    r59_lane = tl.arange($(512))
    r59_value = tl.sqrt(tl.sum(tl.load($(vto_copy_29_ptr) + r59_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_29_ptr) + r59_lane[None, :], dtype=tl.float32), axis=1))
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(1)])
    tl.store($(vlinalg_vector_norm_29_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(512))
    r60_value = tl.load($(input0_ptr) + (r60_z + $(15360)), dtype=tl.bfloat16)
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(512)])
    tl.store($(vto_copy_30_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(1))
    r61_lane = tl.arange($(512))
    r61_value = tl.sqrt(tl.sum(tl.load($(vto_copy_30_ptr) + r61_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_30_ptr) + r61_lane[None, :], dtype=tl.float32), axis=1))
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(1)])
    tl.store($(vlinalg_vector_norm_30_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(512))
    r62_value = tl.load($(input0_ptr) + (r62_z + $(15872)), dtype=tl.bfloat16)
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(512)])
    tl.store($(vto_copy_31_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(1))
    r63_lane = tl.arange($(512))
    r63_value = tl.sqrt(tl.sum(tl.load($(vto_copy_31_ptr) + r63_lane[None, :], dtype=tl.float32) * tl.load($(vto_copy_31_ptr) + r63_lane[None, :], dtype=tl.float32), axis=1))
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(1)])
    tl.store($(vlinalg_vector_norm_31_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(32))
    r64_value = tl.where(r64_z < $(1), tl.load($(vlinalg_vector_norm_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(2), tl.load($(vlinalg_vector_norm_1_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(3), tl.load($(vlinalg_vector_norm_2_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(4), tl.load($(vlinalg_vector_norm_3_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(5), tl.load($(vlinalg_vector_norm_4_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(6), tl.load($(vlinalg_vector_norm_5_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(7), tl.load($(vlinalg_vector_norm_6_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(8), tl.load($(vlinalg_vector_norm_7_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(9), tl.load($(vlinalg_vector_norm_8_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(10), tl.load($(vlinalg_vector_norm_9_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(11), tl.load($(vlinalg_vector_norm_10_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(12), tl.load($(vlinalg_vector_norm_11_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(13), tl.load($(vlinalg_vector_norm_12_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(14), tl.load($(vlinalg_vector_norm_13_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(15), tl.load($(vlinalg_vector_norm_14_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(16), tl.load($(vlinalg_vector_norm_15_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(17), tl.load($(vlinalg_vector_norm_16_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(18), tl.load($(vlinalg_vector_norm_17_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(19), tl.load($(vlinalg_vector_norm_18_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(20), tl.load($(vlinalg_vector_norm_19_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(21), tl.load($(vlinalg_vector_norm_20_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(22), tl.load($(vlinalg_vector_norm_21_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(23), tl.load($(vlinalg_vector_norm_22_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(24), tl.load($(vlinalg_vector_norm_23_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(25), tl.load($(vlinalg_vector_norm_24_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(26), tl.load($(vlinalg_vector_norm_25_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(27), tl.load($(vlinalg_vector_norm_26_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(28), tl.load($(vlinalg_vector_norm_27_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(29), tl.load($(vlinalg_vector_norm_28_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(30), tl.load($(vlinalg_vector_norm_29_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(31), tl.load($(vlinalg_vector_norm_30_ptr) + $(0), dtype=tl.float32), tl.load($(vlinalg_vector_norm_31_ptr) + $(0), dtype=tl.float32))))))))))))))))))))))))))))))))
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(32)])
    tl.store($(vstack_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(32))
    r65_value = tl.load($(vstack_ptr) + r65_z, dtype=tl.float32)
    r65_value = r65_value.to(tl.float64)
    r65_value = tl.broadcast_to(r65_value, [$(32)])
    tl.store($(vto_copy_32_ptr) + r65_z, r65_value)
    z = tl.arange($(32))
    value = tl.load($(vto_copy_32_ptr) + z, dtype=tl.float64)
    value = tl.broadcast_to(value, [$(32)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vlinalg_vector_norm_ptr : RegionName := "vlinalg_vector_norm"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vlinalg_vector_norm_1_ptr : RegionName := "vlinalg_vector_norm_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vlinalg_vector_norm_2_ptr : RegionName := "vlinalg_vector_norm_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vlinalg_vector_norm_3_ptr : RegionName := "vlinalg_vector_norm_3"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vlinalg_vector_norm_4_ptr : RegionName := "vlinalg_vector_norm_4"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vlinalg_vector_norm_5_ptr : RegionName := "vlinalg_vector_norm_5"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vlinalg_vector_norm_6_ptr : RegionName := "vlinalg_vector_norm_6"
  let vto_copy_7_ptr : RegionName := "vto_copy_7"
  let vlinalg_vector_norm_7_ptr : RegionName := "vlinalg_vector_norm_7"
  let vto_copy_8_ptr : RegionName := "vto_copy_8"
  let vlinalg_vector_norm_8_ptr : RegionName := "vlinalg_vector_norm_8"
  let vto_copy_9_ptr : RegionName := "vto_copy_9"
  let vlinalg_vector_norm_9_ptr : RegionName := "vlinalg_vector_norm_9"
  let vto_copy_10_ptr : RegionName := "vto_copy_10"
  let vlinalg_vector_norm_10_ptr : RegionName := "vlinalg_vector_norm_10"
  let vto_copy_11_ptr : RegionName := "vto_copy_11"
  let vlinalg_vector_norm_11_ptr : RegionName := "vlinalg_vector_norm_11"
  let vto_copy_12_ptr : RegionName := "vto_copy_12"
  let vlinalg_vector_norm_12_ptr : RegionName := "vlinalg_vector_norm_12"
  let vto_copy_13_ptr : RegionName := "vto_copy_13"
  let vlinalg_vector_norm_13_ptr : RegionName := "vlinalg_vector_norm_13"
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vlinalg_vector_norm_14_ptr : RegionName := "vlinalg_vector_norm_14"
  let vto_copy_15_ptr : RegionName := "vto_copy_15"
  let vlinalg_vector_norm_15_ptr : RegionName := "vlinalg_vector_norm_15"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vlinalg_vector_norm_16_ptr : RegionName := "vlinalg_vector_norm_16"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vlinalg_vector_norm_17_ptr : RegionName := "vlinalg_vector_norm_17"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vlinalg_vector_norm_18_ptr : RegionName := "vlinalg_vector_norm_18"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vlinalg_vector_norm_19_ptr : RegionName := "vlinalg_vector_norm_19"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vlinalg_vector_norm_20_ptr : RegionName := "vlinalg_vector_norm_20"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vlinalg_vector_norm_21_ptr : RegionName := "vlinalg_vector_norm_21"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vlinalg_vector_norm_22_ptr : RegionName := "vlinalg_vector_norm_22"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vlinalg_vector_norm_23_ptr : RegionName := "vlinalg_vector_norm_23"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vlinalg_vector_norm_24_ptr : RegionName := "vlinalg_vector_norm_24"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vlinalg_vector_norm_25_ptr : RegionName := "vlinalg_vector_norm_25"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vlinalg_vector_norm_26_ptr : RegionName := "vlinalg_vector_norm_26"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vlinalg_vector_norm_27_ptr : RegionName := "vlinalg_vector_norm_27"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vlinalg_vector_norm_28_ptr : RegionName := "vlinalg_vector_norm_28"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vlinalg_vector_norm_29_ptr : RegionName := "vlinalg_vector_norm_29"
  let vto_copy_30_ptr : RegionName := "vto_copy_30"
  let vlinalg_vector_norm_30_ptr : RegionName := "vlinalg_vector_norm_30"
  let vto_copy_31_ptr : RegionName := "vto_copy_31"
  let vlinalg_vector_norm_31_ptr : RegionName := "vlinalg_vector_norm_31"
  let vstack_ptr : RegionName := "vstack"
  let vto_copy_32_ptr : RegionName := "vto_copy_32"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vto_copy_7_ptr : RegionName := "candidate_vto_copy_7"
  let candidate_vto_copy_8_ptr : RegionName := "candidate_vto_copy_8"
  let candidate_vto_copy_9_ptr : RegionName := "candidate_vto_copy_9"
  let candidate_vto_copy_10_ptr : RegionName := "candidate_vto_copy_10"
  let candidate_vto_copy_11_ptr : RegionName := "candidate_vto_copy_11"
  let candidate_vto_copy_12_ptr : RegionName := "candidate_vto_copy_12"
  let candidate_vto_copy_13_ptr : RegionName := "candidate_vto_copy_13"
  let candidate_vto_copy_14_ptr : RegionName := "candidate_vto_copy_14"
  let candidate_vto_copy_15_ptr : RegionName := "candidate_vto_copy_15"
  let candidate_vto_copy_16_ptr : RegionName := "candidate_vto_copy_16"
  let candidate_vto_copy_17_ptr : RegionName := "candidate_vto_copy_17"
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vlinalg_vector_norm_ptr : RegionName := "candidate_vlinalg_vector_norm"
  let candidate_vlinalg_vector_norm_1_ptr : RegionName := "candidate_vlinalg_vector_norm_1"
  let candidate_vlinalg_vector_norm_2_ptr : RegionName := "candidate_vlinalg_vector_norm_2"
  let candidate_vlinalg_vector_norm_3_ptr : RegionName := "candidate_vlinalg_vector_norm_3"
  let candidate_vlinalg_vector_norm_4_ptr : RegionName := "candidate_vlinalg_vector_norm_4"
  let candidate_vlinalg_vector_norm_5_ptr : RegionName := "candidate_vlinalg_vector_norm_5"
  let candidate_vlinalg_vector_norm_6_ptr : RegionName := "candidate_vlinalg_vector_norm_6"
  let candidate_vlinalg_vector_norm_7_ptr : RegionName := "candidate_vlinalg_vector_norm_7"
  let candidate_vlinalg_vector_norm_8_ptr : RegionName := "candidate_vlinalg_vector_norm_8"
  let candidate_vlinalg_vector_norm_9_ptr : RegionName := "candidate_vlinalg_vector_norm_9"
  let candidate_vlinalg_vector_norm_10_ptr : RegionName := "candidate_vlinalg_vector_norm_10"
  let candidate_vlinalg_vector_norm_11_ptr : RegionName := "candidate_vlinalg_vector_norm_11"
  let candidate_vlinalg_vector_norm_12_ptr : RegionName := "candidate_vlinalg_vector_norm_12"
  let candidate_vlinalg_vector_norm_13_ptr : RegionName := "candidate_vlinalg_vector_norm_13"
  let candidate_vlinalg_vector_norm_14_ptr : RegionName := "candidate_vlinalg_vector_norm_14"
  let candidate_vlinalg_vector_norm_15_ptr : RegionName := "candidate_vlinalg_vector_norm_15"
  let candidate_vlinalg_vector_norm_16_ptr : RegionName := "candidate_vlinalg_vector_norm_16"
  let candidate_vlinalg_vector_norm_17_ptr : RegionName := "candidate_vlinalg_vector_norm_17"
  let candidate_vlinalg_vector_norm_18_ptr : RegionName := "candidate_vlinalg_vector_norm_18"
  let candidate_vlinalg_vector_norm_19_ptr : RegionName := "candidate_vlinalg_vector_norm_19"
  let candidate_vlinalg_vector_norm_20_ptr : RegionName := "candidate_vlinalg_vector_norm_20"
  let candidate_vlinalg_vector_norm_21_ptr : RegionName := "candidate_vlinalg_vector_norm_21"
  let candidate_vlinalg_vector_norm_22_ptr : RegionName := "candidate_vlinalg_vector_norm_22"
  let candidate_vlinalg_vector_norm_23_ptr : RegionName := "candidate_vlinalg_vector_norm_23"
  let candidate_vlinalg_vector_norm_24_ptr : RegionName := "candidate_vlinalg_vector_norm_24"
  let candidate_vlinalg_vector_norm_25_ptr : RegionName := "candidate_vlinalg_vector_norm_25"
  let candidate_vlinalg_vector_norm_26_ptr : RegionName := "candidate_vlinalg_vector_norm_26"
  let candidate_vlinalg_vector_norm_27_ptr : RegionName := "candidate_vlinalg_vector_norm_27"
  let candidate_vlinalg_vector_norm_28_ptr : RegionName := "candidate_vlinalg_vector_norm_28"
  let candidate_vlinalg_vector_norm_29_ptr : RegionName := "candidate_vlinalg_vector_norm_29"
  let candidate_vlinalg_vector_norm_30_ptr : RegionName := "candidate_vlinalg_vector_norm_30"
  let candidate_vlinalg_vector_norm_31_ptr : RegionName := "candidate_vlinalg_vector_norm_31"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  triton {
    r0_z = tl.arange($(512))
    r0_value = tl.load($(input0_ptr) + r0_z, dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(512)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(512))
    r1_value = tl.load($(input0_ptr) + (r1_z + $(512)), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(512)])
    tl.store($(candidate_vto_copy_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = tl.load($(input0_ptr) + (r2_z + $(1024)), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(candidate_vto_copy_2_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(512))
    r3_value = tl.load($(input0_ptr) + (r3_z + $(1536)), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(512)])
    tl.store($(candidate_vto_copy_3_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(512))
    r4_value = tl.load($(input0_ptr) + (r4_z + $(2048)), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(512)])
    tl.store($(candidate_vto_copy_4_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = tl.load($(input0_ptr) + (r5_z + $(2560)), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(candidate_vto_copy_5_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(512))
    r6_value = tl.load($(input0_ptr) + (r6_z + $(3072)), dtype=tl.bfloat16)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(512)])
    tl.store($(candidate_vto_copy_6_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(512))
    r7_value = tl.load($(input0_ptr) + (r7_z + $(3584)), dtype=tl.bfloat16)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(512)])
    tl.store($(candidate_vto_copy_7_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = tl.load($(input0_ptr) + (r8_z + $(4096)), dtype=tl.bfloat16)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(candidate_vto_copy_8_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(512))
    r9_value = tl.load($(input0_ptr) + (r9_z + $(4608)), dtype=tl.bfloat16)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(512)])
    tl.store($(candidate_vto_copy_9_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_value = tl.load($(input0_ptr) + (r10_z + $(5120)), dtype=tl.bfloat16)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(candidate_vto_copy_10_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(512))
    r11_value = tl.load($(input0_ptr) + (r11_z + $(5632)), dtype=tl.bfloat16)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(512)])
    tl.store($(candidate_vto_copy_11_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_value = tl.load($(input0_ptr) + (r12_z + $(6144)), dtype=tl.bfloat16)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(candidate_vto_copy_12_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = tl.load($(input0_ptr) + (r13_z + $(6656)), dtype=tl.bfloat16)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(candidate_vto_copy_13_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(512))
    r14_value = tl.load($(input0_ptr) + (r14_z + $(7168)), dtype=tl.bfloat16)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(512)])
    tl.store($(candidate_vto_copy_14_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(512))
    r15_value = tl.load($(input0_ptr) + (r15_z + $(7680)), dtype=tl.bfloat16)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(candidate_vto_copy_15_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(512))
    r16_value = tl.load($(input0_ptr) + (r16_z + $(8192)), dtype=tl.bfloat16)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(512)])
    tl.store($(candidate_vto_copy_16_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(512))
    r17_value = tl.load($(input0_ptr) + (r17_z + $(8704)), dtype=tl.bfloat16)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(512)])
    tl.store($(candidate_vto_copy_17_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(512))
    r18_value = tl.load($(input0_ptr) + (r18_z + $(9216)), dtype=tl.bfloat16)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(512)])
    tl.store($(candidate_vto_copy_18_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(512))
    r19_value = tl.load($(input0_ptr) + (r19_z + $(9728)), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(512)])
    tl.store($(candidate_vto_copy_19_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(512))
    r20_value = tl.load($(input0_ptr) + (r20_z + $(10240)), dtype=tl.bfloat16)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(512)])
    tl.store($(candidate_vto_copy_20_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(512))
    r21_value = tl.load($(input0_ptr) + (r21_z + $(10752)), dtype=tl.bfloat16)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(512)])
    tl.store($(candidate_vto_copy_21_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(512))
    r22_value = tl.load($(input0_ptr) + (r22_z + $(11264)), dtype=tl.bfloat16)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(512)])
    tl.store($(candidate_vto_copy_22_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(512))
    r23_value = tl.load($(input0_ptr) + (r23_z + $(11776)), dtype=tl.bfloat16)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(512)])
    tl.store($(candidate_vto_copy_23_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(512))
    r24_value = tl.load($(input0_ptr) + (r24_z + $(12288)), dtype=tl.bfloat16)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(512)])
    tl.store($(candidate_vto_copy_24_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(512))
    r25_value = tl.load($(input0_ptr) + (r25_z + $(12800)), dtype=tl.bfloat16)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(512)])
    tl.store($(candidate_vto_copy_25_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(512))
    r26_value = tl.load($(input0_ptr) + (r26_z + $(13312)), dtype=tl.bfloat16)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(512)])
    tl.store($(candidate_vto_copy_26_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(512))
    r27_value = tl.load($(input0_ptr) + (r27_z + $(13824)), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(512)])
    tl.store($(candidate_vto_copy_27_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(512))
    r28_value = tl.load($(input0_ptr) + (r28_z + $(14336)), dtype=tl.bfloat16)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(512)])
    tl.store($(candidate_vto_copy_28_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(512))
    r29_value = tl.load($(input0_ptr) + (r29_z + $(14848)), dtype=tl.bfloat16)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(512)])
    tl.store($(candidate_vto_copy_29_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(512))
    r30_value = tl.load($(input0_ptr) + (r30_z + $(15360)), dtype=tl.bfloat16)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(512)])
    tl.store($(candidate_vto_copy_30_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(512))
    r31_value = tl.load($(input0_ptr) + (r31_z + $(15872)), dtype=tl.bfloat16)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(512)])
    tl.store($(candidate_vto_copy_31_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(1))
    r32_lane = tl.arange($(512))
    r32_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_ptr) + r32_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + r32_lane[None, :], dtype=tl.float32), axis=1))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(1))
    r33_lane = tl.arange($(512))
    r33_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_1_ptr) + r33_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_1_ptr) + r33_lane[None, :], dtype=tl.float32), axis=1))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(1))
    r34_lane = tl.arange($(512))
    r34_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_2_ptr) + r34_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_2_ptr) + r34_lane[None, :], dtype=tl.float32), axis=1))
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_2_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(1))
    r35_lane = tl.arange($(512))
    r35_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_3_ptr) + r35_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_3_ptr) + r35_lane[None, :], dtype=tl.float32), axis=1))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_3_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(1))
    r36_lane = tl.arange($(512))
    r36_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_4_ptr) + r36_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_4_ptr) + r36_lane[None, :], dtype=tl.float32), axis=1))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_4_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(1))
    r37_lane = tl.arange($(512))
    r37_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_5_ptr) + r37_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_5_ptr) + r37_lane[None, :], dtype=tl.float32), axis=1))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_5_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(1))
    r38_lane = tl.arange($(512))
    r38_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_6_ptr) + r38_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_6_ptr) + r38_lane[None, :], dtype=tl.float32), axis=1))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_6_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(1))
    r39_lane = tl.arange($(512))
    r39_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_7_ptr) + r39_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_7_ptr) + r39_lane[None, :], dtype=tl.float32), axis=1))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_7_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(1))
    r40_lane = tl.arange($(512))
    r40_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_8_ptr) + r40_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_8_ptr) + r40_lane[None, :], dtype=tl.float32), axis=1))
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_8_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(1))
    r41_lane = tl.arange($(512))
    r41_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_9_ptr) + r41_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_9_ptr) + r41_lane[None, :], dtype=tl.float32), axis=1))
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_9_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(1))
    r42_lane = tl.arange($(512))
    r42_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_10_ptr) + r42_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_10_ptr) + r42_lane[None, :], dtype=tl.float32), axis=1))
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_10_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(1))
    r43_lane = tl.arange($(512))
    r43_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_11_ptr) + r43_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_11_ptr) + r43_lane[None, :], dtype=tl.float32), axis=1))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_11_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(1))
    r44_lane = tl.arange($(512))
    r44_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_12_ptr) + r44_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_12_ptr) + r44_lane[None, :], dtype=tl.float32), axis=1))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_12_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(1))
    r45_lane = tl.arange($(512))
    r45_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_13_ptr) + r45_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_13_ptr) + r45_lane[None, :], dtype=tl.float32), axis=1))
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_13_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(1))
    r46_lane = tl.arange($(512))
    r46_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_14_ptr) + r46_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_14_ptr) + r46_lane[None, :], dtype=tl.float32), axis=1))
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_14_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(1))
    r47_lane = tl.arange($(512))
    r47_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_15_ptr) + r47_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_15_ptr) + r47_lane[None, :], dtype=tl.float32), axis=1))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_15_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(1))
    r48_lane = tl.arange($(512))
    r48_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_16_ptr) + r48_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_16_ptr) + r48_lane[None, :], dtype=tl.float32), axis=1))
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_16_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(1))
    r49_lane = tl.arange($(512))
    r49_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_17_ptr) + r49_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_17_ptr) + r49_lane[None, :], dtype=tl.float32), axis=1))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_17_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(1))
    r50_lane = tl.arange($(512))
    r50_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_18_ptr) + r50_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_18_ptr) + r50_lane[None, :], dtype=tl.float32), axis=1))
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_18_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(1))
    r51_lane = tl.arange($(512))
    r51_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_19_ptr) + r51_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_19_ptr) + r51_lane[None, :], dtype=tl.float32), axis=1))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_19_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(1))
    r52_lane = tl.arange($(512))
    r52_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_20_ptr) + r52_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + r52_lane[None, :], dtype=tl.float32), axis=1))
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_20_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(1))
    r53_lane = tl.arange($(512))
    r53_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_21_ptr) + r53_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_21_ptr) + r53_lane[None, :], dtype=tl.float32), axis=1))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_21_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(1))
    r54_lane = tl.arange($(512))
    r54_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_22_ptr) + r54_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_22_ptr) + r54_lane[None, :], dtype=tl.float32), axis=1))
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_22_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(1))
    r55_lane = tl.arange($(512))
    r55_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_23_ptr) + r55_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_23_ptr) + r55_lane[None, :], dtype=tl.float32), axis=1))
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_23_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(1))
    r56_lane = tl.arange($(512))
    r56_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_24_ptr) + r56_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_24_ptr) + r56_lane[None, :], dtype=tl.float32), axis=1))
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_24_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(1))
    r57_lane = tl.arange($(512))
    r57_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_25_ptr) + r57_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_25_ptr) + r57_lane[None, :], dtype=tl.float32), axis=1))
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_25_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(1))
    r58_lane = tl.arange($(512))
    r58_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_26_ptr) + r58_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_26_ptr) + r58_lane[None, :], dtype=tl.float32), axis=1))
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_26_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(1))
    r59_lane = tl.arange($(512))
    r59_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_27_ptr) + r59_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_27_ptr) + r59_lane[None, :], dtype=tl.float32), axis=1))
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_27_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(1))
    r60_lane = tl.arange($(512))
    r60_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_28_ptr) + r60_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_28_ptr) + r60_lane[None, :], dtype=tl.float32), axis=1))
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_28_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(1))
    r61_lane = tl.arange($(512))
    r61_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_29_ptr) + r61_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_29_ptr) + r61_lane[None, :], dtype=tl.float32), axis=1))
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_29_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(1))
    r62_lane = tl.arange($(512))
    r62_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_30_ptr) + r62_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_30_ptr) + r62_lane[None, :], dtype=tl.float32), axis=1))
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_30_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(1))
    r63_lane = tl.arange($(512))
    r63_value = tl.sqrt(tl.sum(tl.load($(candidate_vto_copy_31_ptr) + r63_lane[None, :], dtype=tl.float32) * tl.load($(candidate_vto_copy_31_ptr) + r63_lane[None, :], dtype=tl.float32), axis=1))
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(1)])
    tl.store($(candidate_vlinalg_vector_norm_31_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(32))
    r64_value = tl.where(r64_z < $(1), tl.load($(candidate_vlinalg_vector_norm_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(2), tl.load($(candidate_vlinalg_vector_norm_1_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(3), tl.load($(candidate_vlinalg_vector_norm_2_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(4), tl.load($(candidate_vlinalg_vector_norm_3_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(5), tl.load($(candidate_vlinalg_vector_norm_4_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(6), tl.load($(candidate_vlinalg_vector_norm_5_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(7), tl.load($(candidate_vlinalg_vector_norm_6_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(8), tl.load($(candidate_vlinalg_vector_norm_7_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(9), tl.load($(candidate_vlinalg_vector_norm_8_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(10), tl.load($(candidate_vlinalg_vector_norm_9_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(11), tl.load($(candidate_vlinalg_vector_norm_10_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(12), tl.load($(candidate_vlinalg_vector_norm_11_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(13), tl.load($(candidate_vlinalg_vector_norm_12_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(14), tl.load($(candidate_vlinalg_vector_norm_13_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(15), tl.load($(candidate_vlinalg_vector_norm_14_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(16), tl.load($(candidate_vlinalg_vector_norm_15_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(17), tl.load($(candidate_vlinalg_vector_norm_16_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(18), tl.load($(candidate_vlinalg_vector_norm_17_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(19), tl.load($(candidate_vlinalg_vector_norm_18_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(20), tl.load($(candidate_vlinalg_vector_norm_19_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(21), tl.load($(candidate_vlinalg_vector_norm_20_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(22), tl.load($(candidate_vlinalg_vector_norm_21_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(23), tl.load($(candidate_vlinalg_vector_norm_22_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(24), tl.load($(candidate_vlinalg_vector_norm_23_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(25), tl.load($(candidate_vlinalg_vector_norm_24_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(26), tl.load($(candidate_vlinalg_vector_norm_25_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(27), tl.load($(candidate_vlinalg_vector_norm_26_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(28), tl.load($(candidate_vlinalg_vector_norm_27_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(29), tl.load($(candidate_vlinalg_vector_norm_28_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(30), tl.load($(candidate_vlinalg_vector_norm_29_ptr) + $(0), dtype=tl.float32), tl.where(r64_z < $(31), tl.load($(candidate_vlinalg_vector_norm_30_ptr) + $(0), dtype=tl.float32), tl.load($(candidate_vlinalg_vector_norm_31_ptr) + $(0), dtype=tl.float32))))))))))))))))))))))))))))))))
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(32)])
    tl.store($(candidate_vstack_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(32))
    r65_value = tl.load($(candidate_vstack_ptr) + r65_z, dtype=tl.float32)
    r65_value = r65_value.to(tl.float64)
    r65_value = tl.broadcast_to(r65_value, [$(32)])
    tl.store($(candidate_vto_copy_32_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(32))
    r66_value = tl.load($(candidate_vto_copy_32_ptr) + ((r66_z - $(0)) % $(32)), dtype=tl.float64)
    r66_value = r66_value.to(tl.float64)
    r66_value = tl.broadcast_to(r66_value, [$(32)])
    tl.store($(candidate_vcat_ptr) + r66_z, r66_value)
    z = tl.arange($(32))
    value = tl.load($(candidate_vcat_ptr) + z, dtype=tl.float64)
    value = tl.broadcast_to(value, [$(32)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_grad_norm_foreach_vs_loop
