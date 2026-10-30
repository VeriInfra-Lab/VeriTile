import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_masked_moe_aux_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let constantf1ebabb642_ptr : RegionName := "constantf1ebabb642"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan_ptr : RegionName := "literal_nan"
  let vge_ptr : RegionName := "vge"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_ptr : RegionName := "vadd"
  let vdiv_ptr : RegionName := "vdiv"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vscatter_ptr : RegionName := "vscatter"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vbitwise_not_ptr : RegionName := "vbitwise_not"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vmul_ptr : RegionName := "vmul"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vfloor_divide_ptr : RegionName := "vfloor_divide"
  let vclamp_min_ptr : RegionName := "vclamp_min"
  let vsum_4_ptr : RegionName := "vsum_4"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vsum_5_ptr : RegionName := "vsum_5"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vneg_ptr : RegionName := "vneg"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vsum_6_ptr : RegionName := "vsum_6"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(32))
    r1_value = (tl.cast((r1_z % $(8)), tl.int64) >= tl.load($(constantf1ebabb642_ptr) + ((r1_z / $(8)) % $(4)), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(32)])
    tl.store($(vge_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = tl.load($(input0_ptr) + ((((r2_z / $(16)) % $(32)) * $(16)) + (r2_z % $(16))), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(vto_copy_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(512))
    r3_value = tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r3_z / $(16)) % $(32)) * $(16)) + (r3_z % $(16))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(512)])
    tl.store($(vsigmoid_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32))
    r4_lane = tl.arange($(16))
    r4_value = tl.sum(tl.load($(vsigmoid_ptr) + ((r4_z[:, None] * $(16)) + r4_lane[None, :]), dtype=tl.float32), axis=1)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32)])
    tl.store($(vsum_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32))
    r5_value = (tl.load($(vsum_1_ptr) + r5_z, dtype=tl.float32) + 0.00000000000000000001)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(32)])
    tl.store($(vadd_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(512))
    r6_value = (tl.load($(vsigmoid_ptr) + ((((r6_z / $(16)) % $(32)) * $(16)) + (r6_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r6_z / $(16)) % $(32)), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(512)])
    tl.store($(vdiv_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(512))
    r7_value = (tl.load($(vdiv_ptr) + ((((r7_z / $(16)) % $(32)) * $(16)) + (r7_z % $(16))), dtype=tl.float32) != tl.load($(vdiv_ptr) + ((((r7_z / $(16)) % $(32)) * $(16)) + (r7_z % $(16))), dtype=tl.float32))
    r7_value = tl.broadcast_to(r7_value, [$(512)])
    tl.store($(visnan_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = tl.where(tl.load($(visnan_ptr) + ((((r8_z / $(16)) % $(32)) * $(16)) + (r8_z % $(16))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(vdiv_ptr) + ((((r8_z / $(16)) % $(32)) * $(16)) + (r8_z % $(16))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(vwhere_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(512))
    r9_lane = tl.arange($(16))
    r9_row_values = tl.load($(vwhere_ptr) + ((((r9_z[:, None] / $(16)) % $(32)) * $(16)) + (r9_lane[None, :] % $(16))), dtype=tl.float32)
    r9_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r9_other in range($(16)) {
      r9_other_value = tl.load($(vwhere_ptr) + ((((r9_z[:, None] / $(16)) % $(32)) * $(16)) + (r9_other % $(16))), dtype=tl.float32)
      r9_other_nan = r9_other_value != r9_other_value
      r9_row_nan = r9_row_values != r9_row_values
      r9_strict = (r9_other_nan & (~r9_row_nan)) | (((~r9_other_nan) & (~r9_row_nan)) & (r9_other_value > r9_row_values))
      r9_equal = (r9_other_nan & r9_row_nan) | (r9_other_value == r9_row_values)
      r9_rank = r9_rank + tl.where(r9_strict | (r9_equal & (r9_other < r9_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r9_selected = r9_rank == tl.cast((r9_z % $(16))[:, None], tl.int64)
    r9_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r9_cutoff in range($(16)) {
      r9_value = r9_value + tl.cast(tl.sum(r9_selected & (r9_lane[None, :] > r9_cutoff), axis=1), tl.int64)
    }
    r9_value = tl.load($(vwhere_ptr) + ((((r9_z / $(16)) % $(32)) * $(16)) + (tl.cast(r9_value, tl.uint32) % $(16))), dtype=tl.float32)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(512)])
    tl.store($(vsort_value_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_lane = tl.arange($(16))
    r10_row_values = tl.load($(vwhere_ptr) + ((((r10_z[:, None] / $(16)) % $(32)) * $(16)) + (r10_lane[None, :] % $(16))), dtype=tl.float32)
    r10_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r10_other in range($(16)) {
      r10_other_value = tl.load($(vwhere_ptr) + ((((r10_z[:, None] / $(16)) % $(32)) * $(16)) + (r10_other % $(16))), dtype=tl.float32)
      r10_other_nan = r10_other_value != r10_other_value
      r10_row_nan = r10_row_values != r10_row_values
      r10_strict = (r10_other_nan & (~r10_row_nan)) | (((~r10_other_nan) & (~r10_row_nan)) & (r10_other_value > r10_row_values))
      r10_equal = (r10_other_nan & r10_row_nan) | (r10_other_value == r10_row_values)
      r10_rank = r10_rank + tl.where(r10_strict | (r10_equal & (r10_other < r10_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r10_selected = r10_rank == tl.cast((r10_z % $(16))[:, None], tl.int64)
    r10_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r10_cutoff in range($(16)) {
      r10_value = r10_value + tl.cast(tl.sum(r10_selected & (r10_lane[None, :] > r10_cutoff), axis=1), tl.int64)
    }
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(vsort_index_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(512))
    r11_value = tl.full([], $(0), dtype=tl.int64)
    r11_value = tl.broadcast_to(r11_value, [$(512)])
    tl.store($(vto_copy_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_value = tl.broadcast_to(tl.load($(vto_copy_1_ptr) + ((((r12_z / $(16)) % $(32)) * $(16)) + (r12_z % $(16))), dtype=tl.int64), [$(512)])
    for r12_lane in range($(2)) {
      r12_value = tl.where(tl.load($(vsort_index_ptr) + ((((r12_z / $(16)) % $(32)) * $(16)) + (r12_lane % $(16))), dtype=tl.int64) == tl.cast((r12_z % $(16)), tl.int64), tl.full([], $(1), dtype=tl.int64), r12_value)
    }
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(vscatter_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = (tl.load($(vscatter_ptr) + ((((r13_z / $(16)) % $(32)) * $(16)) + (r13_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(vto_copy_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(32))
    r14_value = (~tl.load($(vge_ptr) + ((((r14_z / $(8)) % $(4)) * $(8)) + (r14_z % $(8))), dtype=tl.int1))
    r14_value = tl.broadcast_to(r14_value, [$(32)])
    tl.store($(vbitwise_not_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(512))
    r15_value = (tl.load($(vto_copy_2_ptr) + ((((r15_z / $(16)) % $(32)) * $(16)) + (r15_z % $(16))), dtype=tl.int1) & tl.load($(vbitwise_not_ptr) + ((r15_z / $(16)) % $(32)), dtype=tl.int1))
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(vbitwise_and_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(512))
    r16_value = (tl.load($(vdiv_ptr) + ((((r16_z / $(16)) % $(32)) * $(16)) + (r16_z % $(16))), dtype=tl.float32) * tl.where(tl.load($(vbitwise_not_ptr) + ((r16_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(512)])
    tl.store($(vmul_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(16))
    r17_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r17_lane in range($(32)) {
      r17_value = r17_value + tl.where(tl.load($(vbitwise_and_ptr) + ((r17_lane * $(16)) + r17_z), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r17_value = tl.broadcast_to(r17_value, [$(16)])
    tl.store($(vsum_2_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(1))
    r18_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r18_lane in range($(16)) {
      r18_value = r18_value + tl.load($(vsum_2_ptr) + r18_lane, dtype=tl.int64)
    }
    r18_value = tl.broadcast_to(r18_value, [$(1)])
    tl.store($(vsum_3_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(1))
    r19_value = (tl.load($(vsum_3_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r19_value = tl.broadcast_to(r19_value, [$(1)])
    tl.store($(vfloor_divide_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(1))
    r20_value = tl.maximum(tl.load($(vfloor_divide_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r20_value = tl.broadcast_to(r20_value, [$(1)])
    tl.store($(vclamp_min_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(16))
    r21_lane = tl.arange($(32))
    r21_value = tl.sum(tl.load($(vmul_ptr) + ((r21_lane[None, :] * $(16)) + r21_z[:, None]), dtype=tl.float32), axis=1)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(16)])
    tl.store($(vsum_4_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(16))
    r22_value = (tl.load($(vsum_4_ptr) + r22_z, dtype=tl.float32) * tl.toReal(tl.load($(vsum_2_ptr) + r22_z, dtype=tl.int64)))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(16)])
    tl.store($(vmul_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(1))
    r23_lane = tl.arange($(16))
    r23_value = tl.sum(tl.load($(vmul_1_ptr) + r23_lane[None, :], dtype=tl.float32), axis=1)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(1)])
    tl.store($(vsum_5_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(1))
    r24_value = (tl.load($(vclamp_min_ptr) + $(0), dtype=tl.int64) * tl.full([], $(2), dtype=tl.int64))
    r24_value = tl.broadcast_to(r24_value, [$(1)])
    tl.store($(vmul_2_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(1))
    r25_value = (tl.load($(vmul_2_ptr) + $(0), dtype=tl.int64) * tl.load($(vclamp_min_ptr) + $(0), dtype=tl.int64))
    r25_value = tl.broadcast_to(r25_value, [$(1)])
    tl.store($(vmul_3_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(1))
    r26_value = 1.0 / tl.toReal(tl.load($(vmul_3_ptr) + $(0), dtype=tl.int64))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(1)])
    tl.store($(vreciprocal_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(1))
    r27_value = (tl.load($(vreciprocal_ptr) + $(0), dtype=tl.float32) * 2.08)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(1)])
    tl.store($(vmul_4_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(1))
    r28_value = (tl.load($(vsum_5_ptr) + $(0), dtype=tl.float32) * tl.load($(vmul_4_ptr) + $(0), dtype=tl.float32))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(1)])
    tl.store($(vmul_5_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(1))
    r29_value = (1.0 * tl.load($(vmul_4_ptr) + $(0), dtype=tl.float32))
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(1)])
    tl.store($(vmul_6_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(16))
    r30_value = (tl.load($(vmul_6_ptr) + $(0), dtype=tl.float32) * tl.toReal(tl.load($(vsum_2_ptr) + r30_z, dtype=tl.int64)))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(16)])
    tl.store($(vmul_7_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(512))
    r31_value = (tl.load($(vmul_7_ptr) + (r31_z % $(16)), dtype=tl.float32) * tl.where(tl.load($(vbitwise_not_ptr) + ((r31_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(512)])
    tl.store($(vmul_8_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(512))
    r32_value = (tl.load($(vsigmoid_ptr) + ((((r32_z / $(16)) % $(32)) * $(16)) + (r32_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r32_z / $(16)) % $(32)), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(512)])
    tl.store($(vdiv_1_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(512))
    r33_value = (tl.load($(vdiv_1_ptr) + ((((r33_z / $(16)) % $(32)) * $(16)) + (r33_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r33_z / $(16)) % $(32)), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(512)])
    tl.store($(vdiv_2_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(512))
    r34_value = -tl.load($(vmul_8_ptr) + ((((r34_z / $(16)) % $(32)) * $(16)) + (r34_z % $(16))), dtype=tl.float32)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(512)])
    tl.store($(vneg_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(512))
    r35_value = (tl.load($(vneg_ptr) + ((((r35_z / $(16)) % $(32)) * $(16)) + (r35_z % $(16))), dtype=tl.float32) * tl.load($(vdiv_2_ptr) + ((((r35_z / $(16)) % $(32)) * $(16)) + (r35_z % $(16))), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(512)])
    tl.store($(vmul_9_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(512))
    r36_value = (tl.load($(vmul_8_ptr) + ((((r36_z / $(16)) % $(32)) * $(16)) + (r36_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r36_z / $(16)) % $(32)), dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(512)])
    tl.store($(vdiv_3_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(32))
    r37_lane = tl.arange($(16))
    r37_value = tl.sum(tl.load($(vmul_9_ptr) + ((r37_z[:, None] * $(16)) + r37_lane[None, :]), dtype=tl.float32), axis=1)
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(32)])
    tl.store($(vsum_6_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(512))
    r38_value = (tl.load($(vdiv_3_ptr) + ((((r38_z / $(16)) % $(32)) * $(16)) + (r38_z % $(16))), dtype=tl.float32) + tl.load($(vsum_6_ptr) + ((r38_z / $(16)) % $(32)), dtype=tl.float32))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(512)])
    tl.store($(vadd_1_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(512))
    r39_value = (tl.load($(vadd_1_ptr) + ((((r39_z / $(16)) % $(32)) * $(16)) + (r39_z % $(16))), dtype=tl.float32) * tl.load($(vsigmoid_ptr) + ((((r39_z / $(16)) % $(32)) * $(16)) + (r39_z % $(16))), dtype=tl.float32) * ($(1) - tl.load($(vsigmoid_ptr) + ((((r39_z / $(16)) % $(32)) * $(16)) + (r39_z % $(16))), dtype=tl.float32)))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(512)])
    tl.store($(vsigmoid_backward_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(512))
    r40_value = tl.load($(vsigmoid_backward_ptr) + ((((r40_z / $(16)) % $(32)) * $(16)) + (r40_z % $(16))), dtype=tl.float32)
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(512)])
    tl.store($(vto_copy_3_ptr) + r40_z, r40_value)
    z = tl.arange($(512))
    value = tl.load($(vto_copy_3_ptr) + ((((z / $(16)) % $(32)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(512)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let constantf1ebabb642_ptr : RegionName := "constantf1ebabb642"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan_ptr : RegionName := "literal_nan"
  let vge_ptr : RegionName := "vge"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_ptr : RegionName := "vadd"
  let vdiv_ptr : RegionName := "vdiv"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vscatter_ptr : RegionName := "vscatter"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vbitwise_not_ptr : RegionName := "vbitwise_not"
  let vmul_ptr : RegionName := "vmul"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vfloor_divide_ptr : RegionName := "vfloor_divide"
  let vclamp_min_ptr : RegionName := "vclamp_min"
  let vsum_4_ptr : RegionName := "vsum_4"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vsum_5_ptr : RegionName := "vsum_5"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vneg_ptr : RegionName := "vneg"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vsum_6_ptr : RegionName := "vsum_6"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let candidate_literal_nan_ptr : RegionName := "candidate_literal_nan"
  let candidate_vge_ptr : RegionName := "candidate_vge"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vsigmoid_ptr : RegionName := "candidate_vsigmoid"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_visnan_ptr : RegionName := "candidate_visnan"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vsort_value_ptr : RegionName := "candidate_vsort_value"
  let candidate_vsort_index_ptr : RegionName := "candidate_vsort_index"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vscatter_ptr : RegionName := "candidate_vscatter"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vbitwise_not_ptr : RegionName := "candidate_vbitwise_not"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vfloor_divide_ptr : RegionName := "candidate_vfloor_divide"
  let candidate_vclamp_min_ptr : RegionName := "candidate_vclamp_min"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vfloor_divide_1_ptr : RegionName := "candidate_vfloor_divide_1"
  let candidate_vclamp_min_1_ptr : RegionName := "candidate_vclamp_min_1"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vsum_6_ptr : RegionName := "candidate_vsum_6"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vreciprocal_ptr : RegionName := "candidate_vreciprocal"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vmul_6_ptr : RegionName := "candidate_vmul_6"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vsum_7_ptr : RegionName := "candidate_vsum_7"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vsigmoid_backward_ptr : RegionName := "candidate_vsigmoid_backward"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(candidate_literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(32))
    r1_value = (tl.cast((r1_z % $(8)), tl.int64) >= tl.load($(constantf1ebabb642_ptr) + ((r1_z / $(8)) % $(4)), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(32)])
    tl.store($(candidate_vge_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = tl.load($(input0_ptr) + ((((r2_z / $(16)) % $(32)) * $(16)) + (r2_z % $(16))), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(candidate_vto_copy_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(512))
    r3_value = tl.sigmoid(tl.load($(candidate_vto_copy_ptr) + ((((r3_z / $(16)) % $(32)) * $(16)) + (r3_z % $(16))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(512)])
    tl.store($(candidate_vsigmoid_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32))
    r4_lane = tl.arange($(16))
    r4_value = tl.sum(tl.load($(candidate_vsigmoid_ptr) + ((r4_z[:, None] * $(16)) + r4_lane[None, :]), dtype=tl.float32), axis=1)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32)])
    tl.store($(candidate_vsum_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32))
    r5_value = (tl.load($(candidate_vsum_1_ptr) + r5_z, dtype=tl.float32) + 0.00000000000000000001)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(32)])
    tl.store($(candidate_vadd_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(512))
    r6_value = (tl.load($(candidate_vsigmoid_ptr) + ((((r6_z / $(16)) % $(32)) * $(16)) + (r6_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_ptr) + ((r6_z / $(16)) % $(32)), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(512)])
    tl.store($(candidate_vdiv_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(512))
    r7_value = (tl.load($(candidate_vdiv_ptr) + ((((r7_z / $(16)) % $(32)) * $(16)) + (r7_z % $(16))), dtype=tl.float32) != tl.load($(candidate_vdiv_ptr) + ((((r7_z / $(16)) % $(32)) * $(16)) + (r7_z % $(16))), dtype=tl.float32))
    r7_value = tl.broadcast_to(r7_value, [$(512)])
    tl.store($(candidate_visnan_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = tl.where(tl.load($(candidate_visnan_ptr) + ((((r8_z / $(16)) % $(32)) * $(16)) + (r8_z % $(16))), dtype=tl.int1), tl.load($(candidate_literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(candidate_vdiv_ptr) + ((((r8_z / $(16)) % $(32)) * $(16)) + (r8_z % $(16))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(candidate_vwhere_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(512))
    r9_lane = tl.arange($(16))
    r9_row_values = tl.load($(candidate_vwhere_ptr) + ((((r9_z[:, None] / $(16)) % $(32)) * $(16)) + (r9_lane[None, :] % $(16))), dtype=tl.float32)
    r9_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r9_other in range($(16)) {
      r9_other_value = tl.load($(candidate_vwhere_ptr) + ((((r9_z[:, None] / $(16)) % $(32)) * $(16)) + (r9_other % $(16))), dtype=tl.float32)
      r9_other_nan = r9_other_value != r9_other_value
      r9_row_nan = r9_row_values != r9_row_values
      r9_strict = (r9_other_nan & (~r9_row_nan)) | (((~r9_other_nan) & (~r9_row_nan)) & (r9_other_value > r9_row_values))
      r9_equal = (r9_other_nan & r9_row_nan) | (r9_other_value == r9_row_values)
      r9_rank = r9_rank + tl.where(r9_strict | (r9_equal & (r9_other < r9_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r9_selected = r9_rank == tl.cast((r9_z % $(16))[:, None], tl.int64)
    r9_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r9_cutoff in range($(16)) {
      r9_value = r9_value + tl.cast(tl.sum(r9_selected & (r9_lane[None, :] > r9_cutoff), axis=1), tl.int64)
    }
    r9_value = tl.load($(candidate_vwhere_ptr) + ((((r9_z / $(16)) % $(32)) * $(16)) + (tl.cast(r9_value, tl.uint32) % $(16))), dtype=tl.float32)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(512)])
    tl.store($(candidate_vsort_value_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_lane = tl.arange($(16))
    r10_row_values = tl.load($(candidate_vwhere_ptr) + ((((r10_z[:, None] / $(16)) % $(32)) * $(16)) + (r10_lane[None, :] % $(16))), dtype=tl.float32)
    r10_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r10_other in range($(16)) {
      r10_other_value = tl.load($(candidate_vwhere_ptr) + ((((r10_z[:, None] / $(16)) % $(32)) * $(16)) + (r10_other % $(16))), dtype=tl.float32)
      r10_other_nan = r10_other_value != r10_other_value
      r10_row_nan = r10_row_values != r10_row_values
      r10_strict = (r10_other_nan & (~r10_row_nan)) | (((~r10_other_nan) & (~r10_row_nan)) & (r10_other_value > r10_row_values))
      r10_equal = (r10_other_nan & r10_row_nan) | (r10_other_value == r10_row_values)
      r10_rank = r10_rank + tl.where(r10_strict | (r10_equal & (r10_other < r10_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r10_selected = r10_rank == tl.cast((r10_z % $(16))[:, None], tl.int64)
    r10_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r10_cutoff in range($(16)) {
      r10_value = r10_value + tl.cast(tl.sum(r10_selected & (r10_lane[None, :] > r10_cutoff), axis=1), tl.int64)
    }
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(candidate_vsort_index_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(512))
    r11_value = tl.full([], $(0), dtype=tl.int64)
    r11_value = tl.broadcast_to(r11_value, [$(512)])
    tl.store($(candidate_vto_copy_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_value = tl.broadcast_to(tl.load($(candidate_vto_copy_1_ptr) + ((((r12_z / $(16)) % $(32)) * $(16)) + (r12_z % $(16))), dtype=tl.int64), [$(512)])
    for r12_lane in range($(2)) {
      r12_value = tl.where(tl.load($(candidate_vsort_index_ptr) + ((((r12_z / $(16)) % $(32)) * $(16)) + (r12_lane % $(16))), dtype=tl.int64) == tl.cast((r12_z % $(16)), tl.int64), tl.full([], $(1), dtype=tl.int64), r12_value)
    }
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(candidate_vscatter_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = (tl.load($(candidate_vscatter_ptr) + ((((r13_z / $(16)) % $(32)) * $(16)) + (r13_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(candidate_vto_copy_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(32))
    r14_value = (~tl.load($(candidate_vge_ptr) + ((((r14_z / $(8)) % $(4)) * $(8)) + (r14_z % $(8))), dtype=tl.int1))
    r14_value = tl.broadcast_to(r14_value, [$(32)])
    tl.store($(candidate_vbitwise_not_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(512))
    r15_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r15_z / $(16)) % $(32)) * $(16)) + (r15_z % $(16))), dtype=tl.int1) & tl.load($(candidate_vbitwise_not_ptr) + ((r15_z / $(16)) % $(32)), dtype=tl.int1))
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(candidate_vmul_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(512))
    r16_value = (tl.load($(candidate_vdiv_ptr) + ((((r16_z / $(16)) % $(32)) * $(16)) + (r16_z % $(16))), dtype=tl.float32) * tl.where(tl.load($(candidate_vbitwise_not_ptr) + ((r16_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(512)])
    tl.store($(candidate_vmul_1_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(16))
    r17_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r17_lane in range($(32)) {
      r17_value = r17_value + tl.where(tl.load($(candidate_vmul_ptr) + ((r17_lane * $(16)) + r17_z), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r17_value = tl.broadcast_to(r17_value, [$(16)])
    tl.store($(candidate_vsum_2_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(1))
    r18_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r18_lane in range($(16)) {
      r18_value = r18_value + tl.load($(candidate_vsum_2_ptr) + r18_lane, dtype=tl.int64)
    }
    r18_value = tl.broadcast_to(r18_value, [$(1)])
    tl.store($(candidate_vsum_3_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(1))
    r19_value = (tl.load($(candidate_vsum_3_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r19_value = tl.broadcast_to(r19_value, [$(1)])
    tl.store($(candidate_vfloor_divide_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(1))
    r20_value = tl.maximum(tl.load($(candidate_vfloor_divide_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r20_value = tl.broadcast_to(r20_value, [$(1)])
    tl.store($(candidate_vclamp_min_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(1))
    r21_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r21_lane in range($(16)) {
      r21_value = r21_value + tl.load($(candidate_vsum_2_ptr) + r21_lane, dtype=tl.int64)
    }
    r21_value = tl.broadcast_to(r21_value, [$(1)])
    tl.store($(candidate_vsum_4_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(1))
    r22_value = (tl.load($(candidate_vsum_4_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r22_value = tl.broadcast_to(r22_value, [$(1)])
    tl.store($(candidate_vfloor_divide_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(1))
    r23_value = tl.maximum(tl.load($(candidate_vfloor_divide_1_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r23_value = tl.broadcast_to(r23_value, [$(1)])
    tl.store($(candidate_vclamp_min_1_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(16))
    r24_lane = tl.arange($(32))
    r24_value = tl.sum(tl.load($(candidate_vmul_1_ptr) + ((r24_lane[None, :] * $(16)) + r24_z[:, None]), dtype=tl.float32), axis=1)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(16)])
    tl.store($(candidate_vsum_5_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(16))
    r25_value = (tl.load($(candidate_vsum_5_ptr) + r25_z, dtype=tl.float32) * tl.toReal(tl.load($(candidate_vsum_2_ptr) + r25_z, dtype=tl.int64)))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(16)])
    tl.store($(candidate_vmul_2_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(1))
    r26_lane = tl.arange($(16))
    r26_value = tl.sum(tl.load($(candidate_vmul_2_ptr) + r26_lane[None, :], dtype=tl.float32), axis=1)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(1)])
    tl.store($(candidate_vsum_6_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(1))
    r27_value = (tl.load($(candidate_vclamp_min_ptr) + $(0), dtype=tl.int64) * tl.full([], $(2), dtype=tl.int64))
    r27_value = tl.broadcast_to(r27_value, [$(1)])
    tl.store($(candidate_vmul_3_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(1))
    r28_value = (tl.load($(candidate_vmul_3_ptr) + $(0), dtype=tl.int64) * tl.load($(candidate_vclamp_min_1_ptr) + $(0), dtype=tl.int64))
    r28_value = tl.broadcast_to(r28_value, [$(1)])
    tl.store($(candidate_vmul_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(1))
    r29_value = 1.0 / tl.toReal(tl.load($(candidate_vmul_4_ptr) + $(0), dtype=tl.int64))
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(1)])
    tl.store($(candidate_vreciprocal_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(1))
    r30_value = (tl.load($(candidate_vreciprocal_ptr) + $(0), dtype=tl.float32) * 2.08)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(1)])
    tl.store($(candidate_vmul_5_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(1))
    r31_value = (tl.load($(candidate_vsum_6_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vmul_5_ptr) + $(0), dtype=tl.float32))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(1)])
    tl.store($(candidate_vmul_6_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(1))
    r32_value = (1.0 * tl.load($(candidate_vmul_5_ptr) + $(0), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(1)])
    tl.store($(candidate_vmul_7_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(16))
    r33_value = (tl.load($(candidate_vmul_7_ptr) + $(0), dtype=tl.float32) * tl.toReal(tl.load($(candidate_vsum_2_ptr) + r33_z, dtype=tl.int64)))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(16)])
    tl.store($(candidate_vmul_8_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(512))
    r34_value = (tl.load($(candidate_vmul_8_ptr) + (r34_z % $(16)), dtype=tl.float32) * tl.where(tl.load($(candidate_vbitwise_not_ptr) + ((r34_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(512)])
    tl.store($(candidate_vmul_9_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(512))
    r35_value = (tl.load($(candidate_vsigmoid_ptr) + ((((r35_z / $(16)) % $(32)) * $(16)) + (r35_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_ptr) + ((r35_z / $(16)) % $(32)), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(512)])
    tl.store($(candidate_vdiv_1_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(512))
    r36_value = (tl.load($(candidate_vdiv_1_ptr) + ((((r36_z / $(16)) % $(32)) * $(16)) + (r36_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_ptr) + ((r36_z / $(16)) % $(32)), dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(512)])
    tl.store($(candidate_vdiv_2_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(512))
    r37_value = -tl.load($(candidate_vmul_9_ptr) + ((((r37_z / $(16)) % $(32)) * $(16)) + (r37_z % $(16))), dtype=tl.float32)
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(512)])
    tl.store($(candidate_vneg_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(512))
    r38_value = (tl.load($(candidate_vneg_ptr) + ((((r38_z / $(16)) % $(32)) * $(16)) + (r38_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vdiv_2_ptr) + ((((r38_z / $(16)) % $(32)) * $(16)) + (r38_z % $(16))), dtype=tl.float32))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(512)])
    tl.store($(candidate_vmul_10_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(512))
    r39_value = (tl.load($(candidate_vmul_9_ptr) + ((((r39_z / $(16)) % $(32)) * $(16)) + (r39_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_ptr) + ((r39_z / $(16)) % $(32)), dtype=tl.float32))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(512)])
    tl.store($(candidate_vdiv_3_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(32))
    r40_lane = tl.arange($(16))
    r40_value = tl.sum(tl.load($(candidate_vmul_10_ptr) + ((r40_z[:, None] * $(16)) + r40_lane[None, :]), dtype=tl.float32), axis=1)
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(32)])
    tl.store($(candidate_vsum_7_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(512))
    r41_value = (tl.load($(candidate_vdiv_3_ptr) + ((((r41_z / $(16)) % $(32)) * $(16)) + (r41_z % $(16))), dtype=tl.float32) + tl.load($(candidate_vsum_7_ptr) + ((r41_z / $(16)) % $(32)), dtype=tl.float32))
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(512)])
    tl.store($(candidate_vadd_1_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(512))
    r42_value = (tl.load($(candidate_vadd_1_ptr) + ((((r42_z / $(16)) % $(32)) * $(16)) + (r42_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsigmoid_ptr) + ((((r42_z / $(16)) % $(32)) * $(16)) + (r42_z % $(16))), dtype=tl.float32) * ($(1) - tl.load($(candidate_vsigmoid_ptr) + ((((r42_z / $(16)) % $(32)) * $(16)) + (r42_z % $(16))), dtype=tl.float32)))
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(512)])
    tl.store($(candidate_vsigmoid_backward_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(512))
    r43_value = tl.load($(candidate_vsigmoid_backward_ptr) + ((((r43_z / $(16)) % $(32)) * $(16)) + (r43_z % $(16))), dtype=tl.float32)
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(512)])
    tl.store($(candidate_vto_copy_3_ptr) + r43_z, r43_value)
    z = tl.arange($(512))
    value = tl.load($(candidate_vto_copy_3_ptr) + ((((z / $(16)) % $(32)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(512)])
    tl.store($(out_ptr) + z, value)
  }

/-- Finite Real specialization. The non-real NaN literal is assigned the unused
Real payload zero. Its connection to `original` is a separate public goal. -/
def realOriginal : ComputeKernel :=
  let constantf1ebabb642_ptr : RegionName := "constantf1ebabb642"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan_ptr : RegionName := "literal_nan"
  let vge_ptr : RegionName := "vge"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_ptr : RegionName := "vadd"
  let vdiv_ptr : RegionName := "vdiv"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vscatter_ptr : RegionName := "vscatter"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vbitwise_not_ptr : RegionName := "vbitwise_not"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vmul_ptr : RegionName := "vmul"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vfloor_divide_ptr : RegionName := "vfloor_divide"
  let vclamp_min_ptr : RegionName := "vclamp_min"
  let vsum_4_ptr : RegionName := "vsum_4"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vsum_5_ptr : RegionName := "vsum_5"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vneg_ptr : RegionName := "vneg"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vsum_6_ptr : RegionName := "vsum_6"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  triton {
    r0_z = tl.arange($(1))
    r0_value = 0.0
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(32))
    r1_value = (tl.cast((r1_z % $(8)), tl.int64) >= tl.load($(constantf1ebabb642_ptr) + ((r1_z / $(8)) % $(4)), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(32)])
    tl.store($(vge_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = tl.load($(input0_ptr) + ((((r2_z / $(16)) % $(32)) * $(16)) + (r2_z % $(16))), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(vto_copy_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(512))
    r3_value = tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r3_z / $(16)) % $(32)) * $(16)) + (r3_z % $(16))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(512)])
    tl.store($(vsigmoid_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32))
    r4_lane = tl.arange($(16))
    r4_value = tl.sum(tl.load($(vsigmoid_ptr) + ((r4_z[:, None] * $(16)) + r4_lane[None, :]), dtype=tl.float32), axis=1)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32)])
    tl.store($(vsum_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32))
    r5_value = (tl.load($(vsum_1_ptr) + r5_z, dtype=tl.float32) + 0.00000000000000000001)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(32)])
    tl.store($(vadd_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(512))
    r6_value = (tl.load($(vsigmoid_ptr) + ((((r6_z / $(16)) % $(32)) * $(16)) + (r6_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r6_z / $(16)) % $(32)), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(512)])
    tl.store($(vdiv_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(512))
    r7_value = (tl.load($(vdiv_ptr) + ((((r7_z / $(16)) % $(32)) * $(16)) + (r7_z % $(16))), dtype=tl.float32) != tl.load($(vdiv_ptr) + ((((r7_z / $(16)) % $(32)) * $(16)) + (r7_z % $(16))), dtype=tl.float32))
    r7_value = tl.broadcast_to(r7_value, [$(512)])
    tl.store($(visnan_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = tl.where(tl.load($(visnan_ptr) + ((((r8_z / $(16)) % $(32)) * $(16)) + (r8_z % $(16))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(vdiv_ptr) + ((((r8_z / $(16)) % $(32)) * $(16)) + (r8_z % $(16))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(vwhere_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(512))
    r9_lane = tl.arange($(16))
    r9_row_values = tl.load($(vwhere_ptr) + ((((r9_z[:, None] / $(16)) % $(32)) * $(16)) + (r9_lane[None, :] % $(16))), dtype=tl.float32)
    r9_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r9_other in range($(16)) {
      r9_other_value = tl.load($(vwhere_ptr) + ((((r9_z[:, None] / $(16)) % $(32)) * $(16)) + (r9_other % $(16))), dtype=tl.float32)
      r9_other_nan = r9_other_value != r9_other_value
      r9_row_nan = r9_row_values != r9_row_values
      r9_strict = (r9_other_nan & (~r9_row_nan)) | (((~r9_other_nan) & (~r9_row_nan)) & (r9_other_value > r9_row_values))
      r9_equal = (r9_other_nan & r9_row_nan) | (r9_other_value == r9_row_values)
      r9_rank = r9_rank + tl.where(r9_strict | (r9_equal & (r9_other < r9_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r9_selected = r9_rank == tl.cast((r9_z % $(16))[:, None], tl.int64)
    r9_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r9_cutoff in range($(16)) {
      r9_value = r9_value + tl.cast(tl.sum(r9_selected & (r9_lane[None, :] > r9_cutoff), axis=1), tl.int64)
    }
    r9_value = tl.load($(vwhere_ptr) + ((((r9_z / $(16)) % $(32)) * $(16)) + (tl.cast(r9_value, tl.uint32) % $(16))), dtype=tl.float32)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(512)])
    tl.store($(vsort_value_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_lane = tl.arange($(16))
    r10_row_values = tl.load($(vwhere_ptr) + ((((r10_z[:, None] / $(16)) % $(32)) * $(16)) + (r10_lane[None, :] % $(16))), dtype=tl.float32)
    r10_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r10_other in range($(16)) {
      r10_other_value = tl.load($(vwhere_ptr) + ((((r10_z[:, None] / $(16)) % $(32)) * $(16)) + (r10_other % $(16))), dtype=tl.float32)
      r10_other_nan = r10_other_value != r10_other_value
      r10_row_nan = r10_row_values != r10_row_values
      r10_strict = (r10_other_nan & (~r10_row_nan)) | (((~r10_other_nan) & (~r10_row_nan)) & (r10_other_value > r10_row_values))
      r10_equal = (r10_other_nan & r10_row_nan) | (r10_other_value == r10_row_values)
      r10_rank = r10_rank + tl.where(r10_strict | (r10_equal & (r10_other < r10_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r10_selected = r10_rank == tl.cast((r10_z % $(16))[:, None], tl.int64)
    r10_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r10_cutoff in range($(16)) {
      r10_value = r10_value + tl.cast(tl.sum(r10_selected & (r10_lane[None, :] > r10_cutoff), axis=1), tl.int64)
    }
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(vsort_index_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(512))
    r11_value = tl.full([], $(0), dtype=tl.int64)
    r11_value = tl.broadcast_to(r11_value, [$(512)])
    tl.store($(vto_copy_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_value = tl.broadcast_to(tl.load($(vto_copy_1_ptr) + ((((r12_z / $(16)) % $(32)) * $(16)) + (r12_z % $(16))), dtype=tl.int64), [$(512)])
    for r12_lane in range($(2)) {
      r12_value = tl.where(tl.load($(vsort_index_ptr) + ((((r12_z / $(16)) % $(32)) * $(16)) + (r12_lane % $(16))), dtype=tl.int64) == tl.cast((r12_z % $(16)), tl.int64), tl.full([], $(1), dtype=tl.int64), r12_value)
    }
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(vscatter_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = (tl.load($(vscatter_ptr) + ((((r13_z / $(16)) % $(32)) * $(16)) + (r13_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(vto_copy_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(32))
    r14_value = (~tl.load($(vge_ptr) + ((((r14_z / $(8)) % $(4)) * $(8)) + (r14_z % $(8))), dtype=tl.int1))
    r14_value = tl.broadcast_to(r14_value, [$(32)])
    tl.store($(vbitwise_not_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(512))
    r15_value = (tl.load($(vto_copy_2_ptr) + ((((r15_z / $(16)) % $(32)) * $(16)) + (r15_z % $(16))), dtype=tl.int1) & tl.load($(vbitwise_not_ptr) + ((r15_z / $(16)) % $(32)), dtype=tl.int1))
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(vbitwise_and_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(512))
    r16_value = (tl.load($(vdiv_ptr) + ((((r16_z / $(16)) % $(32)) * $(16)) + (r16_z % $(16))), dtype=tl.float32) * tl.where(tl.load($(vbitwise_not_ptr) + ((r16_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(512)])
    tl.store($(vmul_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(16))
    r17_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r17_lane in range($(32)) {
      r17_value = r17_value + tl.where(tl.load($(vbitwise_and_ptr) + ((r17_lane * $(16)) + r17_z), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r17_value = tl.broadcast_to(r17_value, [$(16)])
    tl.store($(vsum_2_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(1))
    r18_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r18_lane in range($(16)) {
      r18_value = r18_value + tl.load($(vsum_2_ptr) + r18_lane, dtype=tl.int64)
    }
    r18_value = tl.broadcast_to(r18_value, [$(1)])
    tl.store($(vsum_3_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(1))
    r19_value = (tl.load($(vsum_3_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r19_value = tl.broadcast_to(r19_value, [$(1)])
    tl.store($(vfloor_divide_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(1))
    r20_value = tl.maximum(tl.load($(vfloor_divide_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r20_value = tl.broadcast_to(r20_value, [$(1)])
    tl.store($(vclamp_min_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(16))
    r21_lane = tl.arange($(32))
    r21_value = tl.sum(tl.load($(vmul_ptr) + ((r21_lane[None, :] * $(16)) + r21_z[:, None]), dtype=tl.float32), axis=1)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(16)])
    tl.store($(vsum_4_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(16))
    r22_value = (tl.load($(vsum_4_ptr) + r22_z, dtype=tl.float32) * tl.toReal(tl.load($(vsum_2_ptr) + r22_z, dtype=tl.int64)))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(16)])
    tl.store($(vmul_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(1))
    r23_lane = tl.arange($(16))
    r23_value = tl.sum(tl.load($(vmul_1_ptr) + r23_lane[None, :], dtype=tl.float32), axis=1)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(1)])
    tl.store($(vsum_5_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(1))
    r24_value = (tl.load($(vclamp_min_ptr) + $(0), dtype=tl.int64) * tl.full([], $(2), dtype=tl.int64))
    r24_value = tl.broadcast_to(r24_value, [$(1)])
    tl.store($(vmul_2_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(1))
    r25_value = (tl.load($(vmul_2_ptr) + $(0), dtype=tl.int64) * tl.load($(vclamp_min_ptr) + $(0), dtype=tl.int64))
    r25_value = tl.broadcast_to(r25_value, [$(1)])
    tl.store($(vmul_3_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(1))
    r26_value = 1.0 / tl.toReal(tl.load($(vmul_3_ptr) + $(0), dtype=tl.int64))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(1)])
    tl.store($(vreciprocal_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(1))
    r27_value = (tl.load($(vreciprocal_ptr) + $(0), dtype=tl.float32) * 2.08)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(1)])
    tl.store($(vmul_4_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(1))
    r28_value = (tl.load($(vsum_5_ptr) + $(0), dtype=tl.float32) * tl.load($(vmul_4_ptr) + $(0), dtype=tl.float32))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(1)])
    tl.store($(vmul_5_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(1))
    r29_value = (1.0 * tl.load($(vmul_4_ptr) + $(0), dtype=tl.float32))
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(1)])
    tl.store($(vmul_6_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(16))
    r30_value = (tl.load($(vmul_6_ptr) + $(0), dtype=tl.float32) * tl.toReal(tl.load($(vsum_2_ptr) + r30_z, dtype=tl.int64)))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(16)])
    tl.store($(vmul_7_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(512))
    r31_value = (tl.load($(vmul_7_ptr) + (r31_z % $(16)), dtype=tl.float32) * tl.where(tl.load($(vbitwise_not_ptr) + ((r31_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(512)])
    tl.store($(vmul_8_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(512))
    r32_value = (tl.load($(vsigmoid_ptr) + ((((r32_z / $(16)) % $(32)) * $(16)) + (r32_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r32_z / $(16)) % $(32)), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(512)])
    tl.store($(vdiv_1_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(512))
    r33_value = (tl.load($(vdiv_1_ptr) + ((((r33_z / $(16)) % $(32)) * $(16)) + (r33_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r33_z / $(16)) % $(32)), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(512)])
    tl.store($(vdiv_2_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(512))
    r34_value = -tl.load($(vmul_8_ptr) + ((((r34_z / $(16)) % $(32)) * $(16)) + (r34_z % $(16))), dtype=tl.float32)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(512)])
    tl.store($(vneg_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(512))
    r35_value = (tl.load($(vneg_ptr) + ((((r35_z / $(16)) % $(32)) * $(16)) + (r35_z % $(16))), dtype=tl.float32) * tl.load($(vdiv_2_ptr) + ((((r35_z / $(16)) % $(32)) * $(16)) + (r35_z % $(16))), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(512)])
    tl.store($(vmul_9_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(512))
    r36_value = (tl.load($(vmul_8_ptr) + ((((r36_z / $(16)) % $(32)) * $(16)) + (r36_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r36_z / $(16)) % $(32)), dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(512)])
    tl.store($(vdiv_3_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(32))
    r37_lane = tl.arange($(16))
    r37_value = tl.sum(tl.load($(vmul_9_ptr) + ((r37_z[:, None] * $(16)) + r37_lane[None, :]), dtype=tl.float32), axis=1)
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(32)])
    tl.store($(vsum_6_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(512))
    r38_value = (tl.load($(vdiv_3_ptr) + ((((r38_z / $(16)) % $(32)) * $(16)) + (r38_z % $(16))), dtype=tl.float32) + tl.load($(vsum_6_ptr) + ((r38_z / $(16)) % $(32)), dtype=tl.float32))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(512)])
    tl.store($(vadd_1_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(512))
    r39_value = (tl.load($(vadd_1_ptr) + ((((r39_z / $(16)) % $(32)) * $(16)) + (r39_z % $(16))), dtype=tl.float32) * tl.load($(vsigmoid_ptr) + ((((r39_z / $(16)) % $(32)) * $(16)) + (r39_z % $(16))), dtype=tl.float32) * ($(1) - tl.load($(vsigmoid_ptr) + ((((r39_z / $(16)) % $(32)) * $(16)) + (r39_z % $(16))), dtype=tl.float32)))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(512)])
    tl.store($(vsigmoid_backward_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(512))
    r40_value = tl.load($(vsigmoid_backward_ptr) + ((((r40_z / $(16)) % $(32)) * $(16)) + (r40_z % $(16))), dtype=tl.float32)
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(512)])
    tl.store($(vto_copy_3_ptr) + r40_z, r40_value)
    z = tl.arange($(512))
    value = tl.load($(vto_copy_3_ptr) + ((((z / $(16)) % $(32)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(512)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_masked_moe_aux_dx
