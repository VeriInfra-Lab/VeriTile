import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_masked_moe_global_aux_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let constantf1ebabb642_ptr : RegionName := "constantf1ebabb642"
  let constant0d12534efc_ptr : RegionName := "constant0d12534efc"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan0_ptr : RegionName := "literal_nan0"
  let literal_nan1_ptr : RegionName := "literal_nan1"
  let vge_ptr : RegionName := "vge"
  let vge_1_ptr : RegionName := "vge_1"
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
  let vsum_2_ptr : RegionName := "vsum_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vsigmoid_1_ptr : RegionName := "vsigmoid_1"
  let vsum_7_ptr : RegionName := "vsum_7"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let visnan_1_ptr : RegionName := "visnan_1"
  let vwhere_1_ptr : RegionName := "vwhere_1"
  let vsort_1_value_ptr : RegionName := "vsort_1_value"
  let vsort_1_index_ptr : RegionName := "vsort_1_index"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vscatter_1_ptr : RegionName := "vscatter_1"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vbitwise_not_1_ptr : RegionName := "vbitwise_not_1"
  let vbitwise_and_1_ptr : RegionName := "vbitwise_and_1"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vsum_8_ptr : RegionName := "vsum_8"
  let vsum_9_ptr : RegionName := "vsum_9"
  let vfloor_divide_2_ptr : RegionName := "vfloor_divide_2"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vsum_10_ptr : RegionName := "vsum_10"
  let vfloor_divide_3_ptr : RegionName := "vfloor_divide_3"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vclamp_min_2_ptr : RegionName := "vclamp_min_2"
  let vdiv_4_ptr : RegionName := "vdiv_4"
  let vdiv_5_ptr : RegionName := "vdiv_5"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vclamp_min_3_ptr : RegionName := "vclamp_min_3"
  let vsum_11_ptr : RegionName := "vsum_11"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vsum_12_ptr : RegionName := "vsum_12"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vreciprocal_1_ptr : RegionName := "vreciprocal_1"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vdiv_6_ptr : RegionName := "vdiv_6"
  let vdiv_7_ptr : RegionName := "vdiv_7"
  let vneg_ptr : RegionName := "vneg"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vdiv_8_ptr : RegionName := "vdiv_8"
  let vsum_13_ptr : RegionName := "vsum_13"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(literal_nan0_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1))
    r1_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(literal_nan1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32))
    r2_value = (tl.cast((r2_z % $(8)), tl.int64) >= tl.load($(constantf1ebabb642_ptr) + ((r2_z / $(8)) % $(4)), dtype=tl.int64))
    r2_value = tl.broadcast_to(r2_value, [$(32)])
    tl.store($(vge_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(32))
    r3_value = (tl.cast((r3_z % $(8)), tl.int64) >= tl.load($(constant0d12534efc_ptr) + ((r3_z / $(8)) % $(4)), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(32)])
    tl.store($(vge_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(512))
    r4_value = tl.load($(input0_ptr) + ((((r4_z / $(16)) % $(32)) * $(16)) + (r4_z % $(16))), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(512)])
    tl.store($(vto_copy_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r5_z / $(16)) % $(32)) * $(16)) + (r5_z % $(16))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(vsigmoid_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32))
    r6_lane = tl.arange($(16))
    r6_value = tl.sum(tl.load($(vsigmoid_ptr) + ((r6_z[:, None] * $(16)) + r6_lane[None, :]), dtype=tl.float32), axis=1)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(32)])
    tl.store($(vsum_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32))
    r7_value = (tl.load($(vsum_1_ptr) + r7_z, dtype=tl.float32) + 0.00000000000000000001)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32)])
    tl.store($(vadd_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = (tl.load($(vsigmoid_ptr) + ((((r8_z / $(16)) % $(32)) * $(16)) + (r8_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r8_z / $(16)) % $(32)), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(vdiv_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(512))
    r9_value = (tl.load($(vdiv_ptr) + ((((r9_z / $(16)) % $(32)) * $(16)) + (r9_z % $(16))), dtype=tl.float32) != tl.load($(vdiv_ptr) + ((((r9_z / $(16)) % $(32)) * $(16)) + (r9_z % $(16))), dtype=tl.float32))
    r9_value = tl.broadcast_to(r9_value, [$(512)])
    tl.store($(visnan_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_value = tl.where(tl.load($(visnan_ptr) + ((((r10_z / $(16)) % $(32)) * $(16)) + (r10_z % $(16))), dtype=tl.int1), tl.load($(literal_nan0_ptr) + $(0), dtype=tl.float32), tl.load($(vdiv_ptr) + ((((r10_z / $(16)) % $(32)) * $(16)) + (r10_z % $(16))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(vwhere_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(512))
    r11_lane = tl.arange($(16))
    r11_row_values = tl.load($(vwhere_ptr) + ((((r11_z[:, None] / $(16)) % $(32)) * $(16)) + (r11_lane[None, :] % $(16))), dtype=tl.float32)
    r11_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r11_other in range($(16)) {
      r11_other_value = tl.load($(vwhere_ptr) + ((((r11_z[:, None] / $(16)) % $(32)) * $(16)) + (r11_other % $(16))), dtype=tl.float32)
      r11_other_nan = r11_other_value != r11_other_value
      r11_row_nan = r11_row_values != r11_row_values
      r11_strict = (r11_other_nan & (~r11_row_nan)) | (((~r11_other_nan) & (~r11_row_nan)) & (r11_other_value > r11_row_values))
      r11_equal = (r11_other_nan & r11_row_nan) | (r11_other_value == r11_row_values)
      r11_rank = r11_rank + tl.where(r11_strict | (r11_equal & (r11_other < r11_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r11_selected = r11_rank == tl.cast((r11_z % $(16))[:, None], tl.int64)
    r11_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r11_cutoff in range($(16)) {
      r11_value = r11_value + tl.cast(tl.sum(r11_selected & (r11_lane[None, :] > r11_cutoff), axis=1), tl.int64)
    }
    r11_value = tl.load($(vwhere_ptr) + ((((r11_z / $(16)) % $(32)) * $(16)) + (tl.cast(r11_value, tl.uint32) % $(16))), dtype=tl.float32)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(512)])
    tl.store($(vsort_value_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_lane = tl.arange($(16))
    r12_row_values = tl.load($(vwhere_ptr) + ((((r12_z[:, None] / $(16)) % $(32)) * $(16)) + (r12_lane[None, :] % $(16))), dtype=tl.float32)
    r12_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r12_other in range($(16)) {
      r12_other_value = tl.load($(vwhere_ptr) + ((((r12_z[:, None] / $(16)) % $(32)) * $(16)) + (r12_other % $(16))), dtype=tl.float32)
      r12_other_nan = r12_other_value != r12_other_value
      r12_row_nan = r12_row_values != r12_row_values
      r12_strict = (r12_other_nan & (~r12_row_nan)) | (((~r12_other_nan) & (~r12_row_nan)) & (r12_other_value > r12_row_values))
      r12_equal = (r12_other_nan & r12_row_nan) | (r12_other_value == r12_row_values)
      r12_rank = r12_rank + tl.where(r12_strict | (r12_equal & (r12_other < r12_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r12_selected = r12_rank == tl.cast((r12_z % $(16))[:, None], tl.int64)
    r12_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r12_cutoff in range($(16)) {
      r12_value = r12_value + tl.cast(tl.sum(r12_selected & (r12_lane[None, :] > r12_cutoff), axis=1), tl.int64)
    }
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(vsort_index_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = tl.full([], $(0), dtype=tl.int64)
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(vto_copy_1_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(512))
    r14_value = tl.broadcast_to(tl.load($(vto_copy_1_ptr) + ((((r14_z / $(16)) % $(32)) * $(16)) + (r14_z % $(16))), dtype=tl.int64), [$(512)])
    for r14_lane in range($(2)) {
      r14_value = tl.where(tl.load($(vsort_index_ptr) + ((((r14_z / $(16)) % $(32)) * $(16)) + (r14_lane % $(16))), dtype=tl.int64) == tl.cast((r14_z % $(16)), tl.int64), tl.full([], $(1), dtype=tl.int64), r14_value)
    }
    r14_value = tl.broadcast_to(r14_value, [$(512)])
    tl.store($(vscatter_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(512))
    r15_value = (tl.load($(vscatter_ptr) + ((((r15_z / $(16)) % $(32)) * $(16)) + (r15_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(vto_copy_2_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(32))
    r16_value = (~tl.load($(vge_ptr) + ((((r16_z / $(8)) % $(4)) * $(8)) + (r16_z % $(8))), dtype=tl.int1))
    r16_value = tl.broadcast_to(r16_value, [$(32)])
    tl.store($(vbitwise_not_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(512))
    r17_value = (tl.load($(vto_copy_2_ptr) + ((((r17_z / $(16)) % $(32)) * $(16)) + (r17_z % $(16))), dtype=tl.int1) & tl.load($(vbitwise_not_ptr) + ((r17_z / $(16)) % $(32)), dtype=tl.int1))
    r17_value = tl.broadcast_to(r17_value, [$(512)])
    tl.store($(vbitwise_and_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(16))
    r18_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r18_lane in range($(32)) {
      r18_value = r18_value + tl.where(tl.load($(vbitwise_and_ptr) + ((r18_lane * $(16)) + r18_z), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r18_value = tl.broadcast_to(r18_value, [$(16)])
    tl.store($(vsum_2_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(512))
    r19_value = tl.load($(input0_ptr) + ((((r19_z / $(16)) % $(32)) * $(16)) + (r19_z % $(16))), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(512)])
    tl.store($(vto_copy_3_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(512))
    r20_value = tl.sigmoid(tl.load($(vto_copy_3_ptr) + ((((r20_z / $(16)) % $(32)) * $(16)) + (r20_z % $(16))), dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(512)])
    tl.store($(vsigmoid_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(32))
    r21_lane = tl.arange($(16))
    r21_value = tl.sum(tl.load($(vsigmoid_1_ptr) + ((r21_z[:, None] * $(16)) + r21_lane[None, :]), dtype=tl.float32), axis=1)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(32)])
    tl.store($(vsum_7_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(32))
    r22_value = (tl.load($(vsum_7_ptr) + r22_z, dtype=tl.float32) + 0.00000000000000000001)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(32)])
    tl.store($(vadd_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(512))
    r23_value = (tl.load($(vsigmoid_1_ptr) + ((((r23_z / $(16)) % $(32)) * $(16)) + (r23_z % $(16))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r23_z / $(16)) % $(32)), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(512)])
    tl.store($(vdiv_3_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(512))
    r24_value = (tl.load($(vdiv_3_ptr) + ((((r24_z / $(16)) % $(32)) * $(16)) + (r24_z % $(16))), dtype=tl.float32) != tl.load($(vdiv_3_ptr) + ((((r24_z / $(16)) % $(32)) * $(16)) + (r24_z % $(16))), dtype=tl.float32))
    r24_value = tl.broadcast_to(r24_value, [$(512)])
    tl.store($(visnan_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(512))
    r25_value = tl.where(tl.load($(visnan_1_ptr) + ((((r25_z / $(16)) % $(32)) * $(16)) + (r25_z % $(16))), dtype=tl.int1), tl.load($(literal_nan1_ptr) + $(0), dtype=tl.float32), tl.load($(vdiv_3_ptr) + ((((r25_z / $(16)) % $(32)) * $(16)) + (r25_z % $(16))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(512)])
    tl.store($(vwhere_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(512))
    r26_lane = tl.arange($(16))
    r26_row_values = tl.load($(vwhere_1_ptr) + ((((r26_z[:, None] / $(16)) % $(32)) * $(16)) + (r26_lane[None, :] % $(16))), dtype=tl.float32)
    r26_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r26_other in range($(16)) {
      r26_other_value = tl.load($(vwhere_1_ptr) + ((((r26_z[:, None] / $(16)) % $(32)) * $(16)) + (r26_other % $(16))), dtype=tl.float32)
      r26_other_nan = r26_other_value != r26_other_value
      r26_row_nan = r26_row_values != r26_row_values
      r26_strict = (r26_other_nan & (~r26_row_nan)) | (((~r26_other_nan) & (~r26_row_nan)) & (r26_other_value > r26_row_values))
      r26_equal = (r26_other_nan & r26_row_nan) | (r26_other_value == r26_row_values)
      r26_rank = r26_rank + tl.where(r26_strict | (r26_equal & (r26_other < r26_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r26_selected = r26_rank == tl.cast((r26_z % $(16))[:, None], tl.int64)
    r26_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r26_cutoff in range($(16)) {
      r26_value = r26_value + tl.cast(tl.sum(r26_selected & (r26_lane[None, :] > r26_cutoff), axis=1), tl.int64)
    }
    r26_value = tl.load($(vwhere_1_ptr) + ((((r26_z / $(16)) % $(32)) * $(16)) + (tl.cast(r26_value, tl.uint32) % $(16))), dtype=tl.float32)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(512)])
    tl.store($(vsort_1_value_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(512))
    r27_lane = tl.arange($(16))
    r27_row_values = tl.load($(vwhere_1_ptr) + ((((r27_z[:, None] / $(16)) % $(32)) * $(16)) + (r27_lane[None, :] % $(16))), dtype=tl.float32)
    r27_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r27_other in range($(16)) {
      r27_other_value = tl.load($(vwhere_1_ptr) + ((((r27_z[:, None] / $(16)) % $(32)) * $(16)) + (r27_other % $(16))), dtype=tl.float32)
      r27_other_nan = r27_other_value != r27_other_value
      r27_row_nan = r27_row_values != r27_row_values
      r27_strict = (r27_other_nan & (~r27_row_nan)) | (((~r27_other_nan) & (~r27_row_nan)) & (r27_other_value > r27_row_values))
      r27_equal = (r27_other_nan & r27_row_nan) | (r27_other_value == r27_row_values)
      r27_rank = r27_rank + tl.where(r27_strict | (r27_equal & (r27_other < r27_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r27_selected = r27_rank == tl.cast((r27_z % $(16))[:, None], tl.int64)
    r27_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r27_cutoff in range($(16)) {
      r27_value = r27_value + tl.cast(tl.sum(r27_selected & (r27_lane[None, :] > r27_cutoff), axis=1), tl.int64)
    }
    r27_value = tl.broadcast_to(r27_value, [$(512)])
    tl.store($(vsort_1_index_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(512))
    r28_value = tl.full([], $(0), dtype=tl.int64)
    r28_value = tl.broadcast_to(r28_value, [$(512)])
    tl.store($(vto_copy_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(512))
    r29_value = tl.broadcast_to(tl.load($(vto_copy_4_ptr) + ((((r29_z / $(16)) % $(32)) * $(16)) + (r29_z % $(16))), dtype=tl.int64), [$(512)])
    for r29_lane in range($(2)) {
      r29_value = tl.where(tl.load($(vsort_1_index_ptr) + ((((r29_z / $(16)) % $(32)) * $(16)) + (r29_lane % $(16))), dtype=tl.int64) == tl.cast((r29_z % $(16)), tl.int64), tl.full([], $(1), dtype=tl.int64), r29_value)
    }
    r29_value = tl.broadcast_to(r29_value, [$(512)])
    tl.store($(vscatter_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(512))
    r30_value = (tl.load($(vscatter_1_ptr) + ((((r30_z / $(16)) % $(32)) * $(16)) + (r30_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r30_value = tl.broadcast_to(r30_value, [$(512)])
    tl.store($(vto_copy_5_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(32))
    r31_value = (~tl.load($(vge_1_ptr) + ((((r31_z / $(8)) % $(4)) * $(8)) + (r31_z % $(8))), dtype=tl.int1))
    r31_value = tl.broadcast_to(r31_value, [$(32)])
    tl.store($(vbitwise_not_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(512))
    r32_value = (tl.load($(vto_copy_5_ptr) + ((((r32_z / $(16)) % $(32)) * $(16)) + (r32_z % $(16))), dtype=tl.int1) & tl.load($(vbitwise_not_1_ptr) + ((r32_z / $(16)) % $(32)), dtype=tl.int1))
    r32_value = tl.broadcast_to(r32_value, [$(512)])
    tl.store($(vbitwise_and_1_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(512))
    r33_value = (tl.load($(vdiv_3_ptr) + ((((r33_z / $(16)) % $(32)) * $(16)) + (r33_z % $(16))), dtype=tl.float32) * tl.where(tl.load($(vbitwise_not_1_ptr) + ((r33_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(512)])
    tl.store($(vmul_8_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(16))
    r34_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r34_lane in range($(32)) {
      r34_value = r34_value + tl.where(tl.load($(vbitwise_and_1_ptr) + ((r34_lane * $(16)) + r34_z), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r34_value = tl.broadcast_to(r34_value, [$(16)])
    tl.store($(vsum_8_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(1))
    r35_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r35_lane in range($(16)) {
      r35_value = r35_value + tl.load($(vsum_8_ptr) + r35_lane, dtype=tl.int64)
    }
    r35_value = tl.broadcast_to(r35_value, [$(1)])
    tl.store($(vsum_9_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(1))
    r36_value = (tl.load($(vsum_9_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r36_value = tl.broadcast_to(r36_value, [$(1)])
    tl.store($(vfloor_divide_2_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(16))
    r37_value = (tl.load($(vsum_2_ptr) + r37_z, dtype=tl.int64) + tl.load($(vsum_8_ptr) + r37_z, dtype=tl.int64))
    r37_value = tl.broadcast_to(r37_value, [$(16)])
    tl.store($(vadd_2_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(1))
    r38_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r38_lane in range($(16)) {
      r38_value = r38_value + tl.load($(vadd_2_ptr) + r38_lane, dtype=tl.int64)
    }
    r38_value = tl.broadcast_to(r38_value, [$(1)])
    tl.store($(vsum_10_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(1))
    r39_value = (tl.load($(vsum_10_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r39_value = tl.broadcast_to(r39_value, [$(1)])
    tl.store($(vfloor_divide_3_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(1))
    r40_value = (tl.full([], $(2), dtype=tl.int64) * tl.load($(vfloor_divide_2_ptr) + $(0), dtype=tl.int64))
    r40_value = tl.broadcast_to(r40_value, [$(1)])
    tl.store($(vmul_9_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(1))
    r41_value = tl.maximum(tl.load($(vfloor_divide_3_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r41_value = tl.broadcast_to(r41_value, [$(1)])
    tl.store($(vclamp_min_2_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(1))
    r42_value = (tl.toReal(tl.load($(vmul_9_ptr) + $(0), dtype=tl.int64)) / tl.toReal(tl.load($(vclamp_min_2_ptr) + $(0), dtype=tl.int64)))
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(1)])
    tl.store($(vdiv_4_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(16))
    r43_value = (tl.toReal(tl.load($(vadd_2_ptr) + r43_z, dtype=tl.int64)) / tl.toReal(tl.full([], $(2), dtype=tl.int64)))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(16)])
    tl.store($(vdiv_5_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(16))
    r44_value = (tl.load($(vdiv_5_ptr) + r44_z, dtype=tl.float32) * tl.load($(vdiv_4_ptr) + $(0), dtype=tl.float32))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(16)])
    tl.store($(vmul_10_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(1))
    r45_value = tl.maximum(tl.load($(vfloor_divide_2_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r45_value = tl.broadcast_to(r45_value, [$(1)])
    tl.store($(vclamp_min_3_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(16))
    r46_lane = tl.arange($(32))
    r46_value = tl.sum(tl.load($(vmul_8_ptr) + ((r46_lane[None, :] * $(16)) + r46_z[:, None]), dtype=tl.float32), axis=1)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(16)])
    tl.store($(vsum_11_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(16))
    r47_value = (tl.load($(vsum_11_ptr) + r47_z, dtype=tl.float32) * tl.load($(vmul_10_ptr) + r47_z, dtype=tl.float32))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(16)])
    tl.store($(vmul_11_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(1))
    r48_lane = tl.arange($(16))
    r48_value = tl.sum(tl.load($(vmul_11_ptr) + r48_lane[None, :], dtype=tl.float32), axis=1)
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(1)])
    tl.store($(vsum_12_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(1))
    r49_value = (tl.load($(vclamp_min_3_ptr) + $(0), dtype=tl.int64) * tl.full([], $(2), dtype=tl.int64))
    r49_value = tl.broadcast_to(r49_value, [$(1)])
    tl.store($(vmul_12_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(1))
    r50_value = (tl.load($(vmul_12_ptr) + $(0), dtype=tl.int64) * tl.load($(vclamp_min_3_ptr) + $(0), dtype=tl.int64))
    r50_value = tl.broadcast_to(r50_value, [$(1)])
    tl.store($(vmul_13_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(1))
    r51_value = 1.0 / tl.toReal(tl.load($(vmul_13_ptr) + $(0), dtype=tl.int64))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(1)])
    tl.store($(vreciprocal_1_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(1))
    r52_value = (tl.load($(vreciprocal_1_ptr) + $(0), dtype=tl.float32) * 2.08)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(1)])
    tl.store($(vmul_14_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(1))
    r53_value = (tl.load($(vsum_12_ptr) + $(0), dtype=tl.float32) * tl.load($(vmul_14_ptr) + $(0), dtype=tl.float32))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(1)])
    tl.store($(vmul_15_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(1))
    r54_value = (1.0 * tl.load($(vmul_14_ptr) + $(0), dtype=tl.float32))
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(1)])
    tl.store($(vmul_16_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(16))
    r55_value = (tl.load($(vmul_16_ptr) + $(0), dtype=tl.float32) * tl.load($(vmul_10_ptr) + r55_z, dtype=tl.float32))
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(16)])
    tl.store($(vmul_17_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(512))
    r56_value = (tl.load($(vmul_17_ptr) + (r56_z % $(16)), dtype=tl.float32) * tl.where(tl.load($(vbitwise_not_1_ptr) + ((r56_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(512)])
    tl.store($(vmul_18_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(512))
    r57_value = (tl.load($(vsigmoid_1_ptr) + ((((r57_z / $(16)) % $(32)) * $(16)) + (r57_z % $(16))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r57_z / $(16)) % $(32)), dtype=tl.float32))
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(512)])
    tl.store($(vdiv_6_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(512))
    r58_value = (tl.load($(vdiv_6_ptr) + ((((r58_z / $(16)) % $(32)) * $(16)) + (r58_z % $(16))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r58_z / $(16)) % $(32)), dtype=tl.float32))
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(512)])
    tl.store($(vdiv_7_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(512))
    r59_value = -tl.load($(vmul_18_ptr) + ((((r59_z / $(16)) % $(32)) * $(16)) + (r59_z % $(16))), dtype=tl.float32)
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(512)])
    tl.store($(vneg_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(512))
    r60_value = (tl.load($(vneg_ptr) + ((((r60_z / $(16)) % $(32)) * $(16)) + (r60_z % $(16))), dtype=tl.float32) * tl.load($(vdiv_7_ptr) + ((((r60_z / $(16)) % $(32)) * $(16)) + (r60_z % $(16))), dtype=tl.float32))
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(512)])
    tl.store($(vmul_19_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(512))
    r61_value = (tl.load($(vmul_18_ptr) + ((((r61_z / $(16)) % $(32)) * $(16)) + (r61_z % $(16))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r61_z / $(16)) % $(32)), dtype=tl.float32))
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(512)])
    tl.store($(vdiv_8_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(32))
    r62_lane = tl.arange($(16))
    r62_value = tl.sum(tl.load($(vmul_19_ptr) + ((r62_z[:, None] * $(16)) + r62_lane[None, :]), dtype=tl.float32), axis=1)
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(32)])
    tl.store($(vsum_13_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(512))
    r63_value = (tl.load($(vdiv_8_ptr) + ((((r63_z / $(16)) % $(32)) * $(16)) + (r63_z % $(16))), dtype=tl.float32) + tl.load($(vsum_13_ptr) + ((r63_z / $(16)) % $(32)), dtype=tl.float32))
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(512)])
    tl.store($(vadd_3_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(512))
    r64_value = (tl.load($(vadd_3_ptr) + ((((r64_z / $(16)) % $(32)) * $(16)) + (r64_z % $(16))), dtype=tl.float32) * tl.load($(vsigmoid_1_ptr) + ((((r64_z / $(16)) % $(32)) * $(16)) + (r64_z % $(16))), dtype=tl.float32) * (1.0 - tl.load($(vsigmoid_1_ptr) + ((((r64_z / $(16)) % $(32)) * $(16)) + (r64_z % $(16))), dtype=tl.float32)))
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(512)])
    tl.store($(vsigmoid_backward_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(512))
    r65_value = tl.load($(vsigmoid_backward_ptr) + ((((r65_z / $(16)) % $(32)) * $(16)) + (r65_z % $(16))), dtype=tl.float32)
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(512)])
    tl.store($(vto_copy_6_ptr) + r65_z, r65_value)
    z = tl.arange($(512))
    value = tl.load($(vto_copy_6_ptr) + ((((z / $(16)) % $(32)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(512)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let constantf1ebabb642_ptr : RegionName := "constantf1ebabb642"
  let constant0d12534efc_ptr : RegionName := "constant0d12534efc"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan0_ptr : RegionName := "literal_nan0"
  let literal_nan1_ptr : RegionName := "literal_nan1"
  let vge_ptr : RegionName := "vge"
  let vge_1_ptr : RegionName := "vge_1"
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
  let vsum_2_ptr : RegionName := "vsum_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vsigmoid_1_ptr : RegionName := "vsigmoid_1"
  let vsum_7_ptr : RegionName := "vsum_7"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let visnan_1_ptr : RegionName := "visnan_1"
  let vwhere_1_ptr : RegionName := "vwhere_1"
  let vsort_1_value_ptr : RegionName := "vsort_1_value"
  let vsort_1_index_ptr : RegionName := "vsort_1_index"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vscatter_1_ptr : RegionName := "vscatter_1"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vbitwise_not_1_ptr : RegionName := "vbitwise_not_1"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vsum_8_ptr : RegionName := "vsum_8"
  let vsum_9_ptr : RegionName := "vsum_9"
  let vfloor_divide_2_ptr : RegionName := "vfloor_divide_2"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vsum_10_ptr : RegionName := "vsum_10"
  let vfloor_divide_3_ptr : RegionName := "vfloor_divide_3"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vclamp_min_2_ptr : RegionName := "vclamp_min_2"
  let vdiv_4_ptr : RegionName := "vdiv_4"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vclamp_min_3_ptr : RegionName := "vclamp_min_3"
  let vsum_11_ptr : RegionName := "vsum_11"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vsum_12_ptr : RegionName := "vsum_12"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vreciprocal_1_ptr : RegionName := "vreciprocal_1"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vneg_ptr : RegionName := "vneg"
  let vsum_13_ptr : RegionName := "vsum_13"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let candidate_literal_nan0_ptr : RegionName := "candidate_literal_nan0"
  let candidate_literal_nan1_ptr : RegionName := "candidate_literal_nan1"
  let candidate_vge_ptr : RegionName := "candidate_vge"
  let candidate_vge_1_ptr : RegionName := "candidate_vge_1"
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
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vsigmoid_1_ptr : RegionName := "candidate_vsigmoid_1"
  let candidate_vsum_7_ptr : RegionName := "candidate_vsum_7"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_visnan_1_ptr : RegionName := "candidate_visnan_1"
  let candidate_vwhere_1_ptr : RegionName := "candidate_vwhere_1"
  let candidate_vsort_1_value_ptr : RegionName := "candidate_vsort_1_value"
  let candidate_vsort_1_index_ptr : RegionName := "candidate_vsort_1_index"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vscatter_1_ptr : RegionName := "candidate_vscatter_1"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vbitwise_not_1_ptr : RegionName := "candidate_vbitwise_not_1"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vsum_8_ptr : RegionName := "candidate_vsum_8"
  let candidate_vsum_9_ptr : RegionName := "candidate_vsum_9"
  let candidate_vfloor_divide_2_ptr : RegionName := "candidate_vfloor_divide_2"
  let candidate_vclamp_min_2_ptr : RegionName := "candidate_vclamp_min_2"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vsum_10_ptr : RegionName := "candidate_vsum_10"
  let candidate_vfloor_divide_3_ptr : RegionName := "candidate_vfloor_divide_3"
  let candidate_vclamp_min_3_ptr : RegionName := "candidate_vclamp_min_3"
  let candidate_vsum_11_ptr : RegionName := "candidate_vsum_11"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vsum_12_ptr : RegionName := "candidate_vsum_12"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vreciprocal_1_ptr : RegionName := "candidate_vreciprocal_1"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vdiv_4_ptr : RegionName := "candidate_vdiv_4"
  let candidate_vsum_13_ptr : RegionName := "candidate_vsum_13"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vsigmoid_backward_ptr : RegionName := "candidate_vsigmoid_backward"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(candidate_literal_nan0_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1))
    r1_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(candidate_literal_nan1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32))
    r2_value = (tl.cast((r2_z % $(8)), tl.int64) >= tl.load($(constantf1ebabb642_ptr) + ((r2_z / $(8)) % $(4)), dtype=tl.int64))
    r2_value = tl.broadcast_to(r2_value, [$(32)])
    tl.store($(candidate_vge_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(32))
    r3_value = (tl.cast((r3_z % $(8)), tl.int64) >= tl.load($(constant0d12534efc_ptr) + ((r3_z / $(8)) % $(4)), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(32)])
    tl.store($(candidate_vge_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(512))
    r4_value = tl.load($(input0_ptr) + ((((r4_z / $(16)) % $(32)) * $(16)) + (r4_z % $(16))), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(512)])
    tl.store($(candidate_vto_copy_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = tl.sigmoid(tl.load($(candidate_vto_copy_ptr) + ((((r5_z / $(16)) % $(32)) * $(16)) + (r5_z % $(16))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(candidate_vsigmoid_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32))
    r6_lane = tl.arange($(16))
    r6_value = tl.sum(tl.load($(candidate_vsigmoid_ptr) + ((r6_z[:, None] * $(16)) + r6_lane[None, :]), dtype=tl.float32), axis=1)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(32)])
    tl.store($(candidate_vsum_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32))
    r7_value = (tl.load($(candidate_vsum_1_ptr) + r7_z, dtype=tl.float32) + 0.00000000000000000001)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32)])
    tl.store($(candidate_vadd_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = (tl.load($(candidate_vsigmoid_ptr) + ((((r8_z / $(16)) % $(32)) * $(16)) + (r8_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_ptr) + ((r8_z / $(16)) % $(32)), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(candidate_vdiv_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(512))
    r9_value = (tl.load($(candidate_vdiv_ptr) + ((((r9_z / $(16)) % $(32)) * $(16)) + (r9_z % $(16))), dtype=tl.float32) != tl.load($(candidate_vdiv_ptr) + ((((r9_z / $(16)) % $(32)) * $(16)) + (r9_z % $(16))), dtype=tl.float32))
    r9_value = tl.broadcast_to(r9_value, [$(512)])
    tl.store($(candidate_visnan_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_value = tl.where(tl.load($(candidate_visnan_ptr) + ((((r10_z / $(16)) % $(32)) * $(16)) + (r10_z % $(16))), dtype=tl.int1), tl.load($(candidate_literal_nan0_ptr) + $(0), dtype=tl.float32), tl.load($(candidate_vdiv_ptr) + ((((r10_z / $(16)) % $(32)) * $(16)) + (r10_z % $(16))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(candidate_vwhere_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(512))
    r11_lane = tl.arange($(16))
    r11_row_values = tl.load($(candidate_vwhere_ptr) + ((((r11_z[:, None] / $(16)) % $(32)) * $(16)) + (r11_lane[None, :] % $(16))), dtype=tl.float32)
    r11_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r11_other in range($(16)) {
      r11_other_value = tl.load($(candidate_vwhere_ptr) + ((((r11_z[:, None] / $(16)) % $(32)) * $(16)) + (r11_other % $(16))), dtype=tl.float32)
      r11_other_nan = r11_other_value != r11_other_value
      r11_row_nan = r11_row_values != r11_row_values
      r11_strict = (r11_other_nan & (~r11_row_nan)) | (((~r11_other_nan) & (~r11_row_nan)) & (r11_other_value > r11_row_values))
      r11_equal = (r11_other_nan & r11_row_nan) | (r11_other_value == r11_row_values)
      r11_rank = r11_rank + tl.where(r11_strict | (r11_equal & (r11_other < r11_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r11_selected = r11_rank == tl.cast((r11_z % $(16))[:, None], tl.int64)
    r11_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r11_cutoff in range($(16)) {
      r11_value = r11_value + tl.cast(tl.sum(r11_selected & (r11_lane[None, :] > r11_cutoff), axis=1), tl.int64)
    }
    r11_value = tl.load($(candidate_vwhere_ptr) + ((((r11_z / $(16)) % $(32)) * $(16)) + (tl.cast(r11_value, tl.uint32) % $(16))), dtype=tl.float32)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(512)])
    tl.store($(candidate_vsort_value_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_lane = tl.arange($(16))
    r12_row_values = tl.load($(candidate_vwhere_ptr) + ((((r12_z[:, None] / $(16)) % $(32)) * $(16)) + (r12_lane[None, :] % $(16))), dtype=tl.float32)
    r12_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r12_other in range($(16)) {
      r12_other_value = tl.load($(candidate_vwhere_ptr) + ((((r12_z[:, None] / $(16)) % $(32)) * $(16)) + (r12_other % $(16))), dtype=tl.float32)
      r12_other_nan = r12_other_value != r12_other_value
      r12_row_nan = r12_row_values != r12_row_values
      r12_strict = (r12_other_nan & (~r12_row_nan)) | (((~r12_other_nan) & (~r12_row_nan)) & (r12_other_value > r12_row_values))
      r12_equal = (r12_other_nan & r12_row_nan) | (r12_other_value == r12_row_values)
      r12_rank = r12_rank + tl.where(r12_strict | (r12_equal & (r12_other < r12_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r12_selected = r12_rank == tl.cast((r12_z % $(16))[:, None], tl.int64)
    r12_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r12_cutoff in range($(16)) {
      r12_value = r12_value + tl.cast(tl.sum(r12_selected & (r12_lane[None, :] > r12_cutoff), axis=1), tl.int64)
    }
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(candidate_vsort_index_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = tl.full([], $(0), dtype=tl.int64)
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(candidate_vto_copy_1_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(512))
    r14_value = tl.broadcast_to(tl.load($(candidate_vto_copy_1_ptr) + ((((r14_z / $(16)) % $(32)) * $(16)) + (r14_z % $(16))), dtype=tl.int64), [$(512)])
    for r14_lane in range($(2)) {
      r14_value = tl.where(tl.load($(candidate_vsort_index_ptr) + ((((r14_z / $(16)) % $(32)) * $(16)) + (r14_lane % $(16))), dtype=tl.int64) == tl.cast((r14_z % $(16)), tl.int64), tl.full([], $(1), dtype=tl.int64), r14_value)
    }
    r14_value = tl.broadcast_to(r14_value, [$(512)])
    tl.store($(candidate_vscatter_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(512))
    r15_value = (tl.load($(candidate_vscatter_ptr) + ((((r15_z / $(16)) % $(32)) * $(16)) + (r15_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(candidate_vto_copy_2_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(32))
    r16_value = (~tl.load($(candidate_vge_ptr) + ((((r16_z / $(8)) % $(4)) * $(8)) + (r16_z % $(8))), dtype=tl.int1))
    r16_value = tl.broadcast_to(r16_value, [$(32)])
    tl.store($(candidate_vbitwise_not_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(512))
    r17_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r17_z / $(16)) % $(32)) * $(16)) + (r17_z % $(16))), dtype=tl.int1) & tl.load($(candidate_vbitwise_not_ptr) + ((r17_z / $(16)) % $(32)), dtype=tl.int1))
    r17_value = tl.broadcast_to(r17_value, [$(512)])
    tl.store($(candidate_vmul_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(16))
    r18_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r18_lane in range($(32)) {
      r18_value = r18_value + tl.where(tl.load($(candidate_vmul_ptr) + ((r18_lane * $(16)) + r18_z), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r18_value = tl.broadcast_to(r18_value, [$(16)])
    tl.store($(candidate_vsum_2_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(512))
    r19_value = tl.load($(input0_ptr) + ((((r19_z / $(16)) % $(32)) * $(16)) + (r19_z % $(16))), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(512)])
    tl.store($(candidate_vto_copy_3_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(512))
    r20_value = tl.sigmoid(tl.load($(candidate_vto_copy_3_ptr) + ((((r20_z / $(16)) % $(32)) * $(16)) + (r20_z % $(16))), dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(512)])
    tl.store($(candidate_vsigmoid_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(32))
    r21_lane = tl.arange($(16))
    r21_value = tl.sum(tl.load($(candidate_vsigmoid_1_ptr) + ((r21_z[:, None] * $(16)) + r21_lane[None, :]), dtype=tl.float32), axis=1)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(32)])
    tl.store($(candidate_vsum_7_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(32))
    r22_value = (tl.load($(candidate_vsum_7_ptr) + r22_z, dtype=tl.float32) + 0.00000000000000000001)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(32)])
    tl.store($(candidate_vadd_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(512))
    r23_value = (tl.load($(candidate_vsigmoid_1_ptr) + ((((r23_z / $(16)) % $(32)) * $(16)) + (r23_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_1_ptr) + ((r23_z / $(16)) % $(32)), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(512)])
    tl.store($(candidate_vdiv_1_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(512))
    r24_value = (tl.load($(candidate_vdiv_1_ptr) + ((((r24_z / $(16)) % $(32)) * $(16)) + (r24_z % $(16))), dtype=tl.float32) != tl.load($(candidate_vdiv_1_ptr) + ((((r24_z / $(16)) % $(32)) * $(16)) + (r24_z % $(16))), dtype=tl.float32))
    r24_value = tl.broadcast_to(r24_value, [$(512)])
    tl.store($(candidate_visnan_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(512))
    r25_value = tl.where(tl.load($(candidate_visnan_1_ptr) + ((((r25_z / $(16)) % $(32)) * $(16)) + (r25_z % $(16))), dtype=tl.int1), tl.load($(candidate_literal_nan1_ptr) + $(0), dtype=tl.float32), tl.load($(candidate_vdiv_1_ptr) + ((((r25_z / $(16)) % $(32)) * $(16)) + (r25_z % $(16))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(512)])
    tl.store($(candidate_vwhere_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(512))
    r26_lane = tl.arange($(16))
    r26_row_values = tl.load($(candidate_vwhere_1_ptr) + ((((r26_z[:, None] / $(16)) % $(32)) * $(16)) + (r26_lane[None, :] % $(16))), dtype=tl.float32)
    r26_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r26_other in range($(16)) {
      r26_other_value = tl.load($(candidate_vwhere_1_ptr) + ((((r26_z[:, None] / $(16)) % $(32)) * $(16)) + (r26_other % $(16))), dtype=tl.float32)
      r26_other_nan = r26_other_value != r26_other_value
      r26_row_nan = r26_row_values != r26_row_values
      r26_strict = (r26_other_nan & (~r26_row_nan)) | (((~r26_other_nan) & (~r26_row_nan)) & (r26_other_value > r26_row_values))
      r26_equal = (r26_other_nan & r26_row_nan) | (r26_other_value == r26_row_values)
      r26_rank = r26_rank + tl.where(r26_strict | (r26_equal & (r26_other < r26_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r26_selected = r26_rank == tl.cast((r26_z % $(16))[:, None], tl.int64)
    r26_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r26_cutoff in range($(16)) {
      r26_value = r26_value + tl.cast(tl.sum(r26_selected & (r26_lane[None, :] > r26_cutoff), axis=1), tl.int64)
    }
    r26_value = tl.load($(candidate_vwhere_1_ptr) + ((((r26_z / $(16)) % $(32)) * $(16)) + (tl.cast(r26_value, tl.uint32) % $(16))), dtype=tl.float32)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(512)])
    tl.store($(candidate_vsort_1_value_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(512))
    r27_lane = tl.arange($(16))
    r27_row_values = tl.load($(candidate_vwhere_1_ptr) + ((((r27_z[:, None] / $(16)) % $(32)) * $(16)) + (r27_lane[None, :] % $(16))), dtype=tl.float32)
    r27_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r27_other in range($(16)) {
      r27_other_value = tl.load($(candidate_vwhere_1_ptr) + ((((r27_z[:, None] / $(16)) % $(32)) * $(16)) + (r27_other % $(16))), dtype=tl.float32)
      r27_other_nan = r27_other_value != r27_other_value
      r27_row_nan = r27_row_values != r27_row_values
      r27_strict = (r27_other_nan & (~r27_row_nan)) | (((~r27_other_nan) & (~r27_row_nan)) & (r27_other_value > r27_row_values))
      r27_equal = (r27_other_nan & r27_row_nan) | (r27_other_value == r27_row_values)
      r27_rank = r27_rank + tl.where(r27_strict | (r27_equal & (r27_other < r27_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r27_selected = r27_rank == tl.cast((r27_z % $(16))[:, None], tl.int64)
    r27_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r27_cutoff in range($(16)) {
      r27_value = r27_value + tl.cast(tl.sum(r27_selected & (r27_lane[None, :] > r27_cutoff), axis=1), tl.int64)
    }
    r27_value = tl.broadcast_to(r27_value, [$(512)])
    tl.store($(candidate_vsort_1_index_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(512))
    r28_value = tl.full([], $(0), dtype=tl.int64)
    r28_value = tl.broadcast_to(r28_value, [$(512)])
    tl.store($(candidate_vto_copy_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(512))
    r29_value = tl.broadcast_to(tl.load($(candidate_vto_copy_4_ptr) + ((((r29_z / $(16)) % $(32)) * $(16)) + (r29_z % $(16))), dtype=tl.int64), [$(512)])
    for r29_lane in range($(2)) {
      r29_value = tl.where(tl.load($(candidate_vsort_1_index_ptr) + ((((r29_z / $(16)) % $(32)) * $(16)) + (r29_lane % $(16))), dtype=tl.int64) == tl.cast((r29_z % $(16)), tl.int64), tl.full([], $(1), dtype=tl.int64), r29_value)
    }
    r29_value = tl.broadcast_to(r29_value, [$(512)])
    tl.store($(candidate_vscatter_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(512))
    r30_value = (tl.load($(candidate_vscatter_1_ptr) + ((((r30_z / $(16)) % $(32)) * $(16)) + (r30_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r30_value = tl.broadcast_to(r30_value, [$(512)])
    tl.store($(candidate_vto_copy_5_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(32))
    r31_value = (~tl.load($(candidate_vge_1_ptr) + ((((r31_z / $(8)) % $(4)) * $(8)) + (r31_z % $(8))), dtype=tl.int1))
    r31_value = tl.broadcast_to(r31_value, [$(32)])
    tl.store($(candidate_vbitwise_not_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(512))
    r32_value = (tl.load($(candidate_vto_copy_5_ptr) + ((((r32_z / $(16)) % $(32)) * $(16)) + (r32_z % $(16))), dtype=tl.int1) & tl.load($(candidate_vbitwise_not_1_ptr) + ((r32_z / $(16)) % $(32)), dtype=tl.int1))
    r32_value = tl.broadcast_to(r32_value, [$(512)])
    tl.store($(candidate_vmul_7_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(512))
    r33_value = (tl.load($(candidate_vdiv_1_ptr) + ((((r33_z / $(16)) % $(32)) * $(16)) + (r33_z % $(16))), dtype=tl.float32) * tl.where(tl.load($(candidate_vbitwise_not_1_ptr) + ((r33_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(512)])
    tl.store($(candidate_vmul_8_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(16))
    r34_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r34_lane in range($(32)) {
      r34_value = r34_value + tl.where(tl.load($(candidate_vmul_7_ptr) + ((r34_lane * $(16)) + r34_z), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r34_value = tl.broadcast_to(r34_value, [$(16)])
    tl.store($(candidate_vsum_8_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(1))
    r35_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r35_lane in range($(16)) {
      r35_value = r35_value + tl.load($(candidate_vsum_8_ptr) + r35_lane, dtype=tl.int64)
    }
    r35_value = tl.broadcast_to(r35_value, [$(1)])
    tl.store($(candidate_vsum_9_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(1))
    r36_value = (tl.load($(candidate_vsum_9_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r36_value = tl.broadcast_to(r36_value, [$(1)])
    tl.store($(candidate_vfloor_divide_2_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(1))
    r37_value = tl.maximum(tl.load($(candidate_vfloor_divide_2_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r37_value = tl.broadcast_to(r37_value, [$(1)])
    tl.store($(candidate_vclamp_min_2_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(16))
    r38_value = (tl.load($(candidate_vsum_2_ptr) + r38_z, dtype=tl.int64) + tl.load($(candidate_vsum_8_ptr) + r38_z, dtype=tl.int64))
    r38_value = tl.broadcast_to(r38_value, [$(16)])
    tl.store($(candidate_vadd_2_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(1))
    r39_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r39_lane in range($(16)) {
      r39_value = r39_value + tl.load($(candidate_vadd_2_ptr) + r39_lane, dtype=tl.int64)
    }
    r39_value = tl.broadcast_to(r39_value, [$(1)])
    tl.store($(candidate_vsum_10_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(1))
    r40_value = (tl.load($(candidate_vsum_10_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r40_value = tl.broadcast_to(r40_value, [$(1)])
    tl.store($(candidate_vfloor_divide_3_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(1))
    r41_value = tl.maximum(tl.load($(candidate_vfloor_divide_3_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r41_value = tl.broadcast_to(r41_value, [$(1)])
    tl.store($(candidate_vclamp_min_3_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(16))
    r42_lane = tl.arange($(32))
    r42_value = tl.sum(tl.load($(candidate_vmul_8_ptr) + ((r42_lane[None, :] * $(16)) + r42_z[:, None]), dtype=tl.float32), axis=1)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(16)])
    tl.store($(candidate_vsum_11_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(16))
    r43_value = (tl.load($(candidate_vsum_11_ptr) + r43_z, dtype=tl.float32) * tl.toReal(tl.load($(candidate_vadd_2_ptr) + r43_z, dtype=tl.int64)))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(16)])
    tl.store($(candidate_vmul_9_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(1))
    r44_lane = tl.arange($(16))
    r44_value = tl.sum(tl.load($(candidate_vmul_9_ptr) + r44_lane[None, :], dtype=tl.float32), axis=1)
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(1)])
    tl.store($(candidate_vsum_12_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(1))
    r45_value = (tl.load($(candidate_vclamp_min_2_ptr) + $(0), dtype=tl.int64) * tl.full([], $(2), dtype=tl.int64))
    r45_value = tl.broadcast_to(r45_value, [$(1)])
    tl.store($(candidate_vmul_10_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(1))
    r46_value = (tl.load($(candidate_vmul_10_ptr) + $(0), dtype=tl.int64) * tl.load($(candidate_vclamp_min_3_ptr) + $(0), dtype=tl.int64))
    r46_value = tl.broadcast_to(r46_value, [$(1)])
    tl.store($(candidate_vmul_11_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(1))
    r47_value = 1.0 / tl.toReal(tl.load($(candidate_vmul_11_ptr) + $(0), dtype=tl.int64))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(1)])
    tl.store($(candidate_vreciprocal_1_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(1))
    r48_value = (tl.load($(candidate_vreciprocal_1_ptr) + $(0), dtype=tl.float32) * 2.08)
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(1)])
    tl.store($(candidate_vmul_12_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(1))
    r49_value = (tl.load($(candidate_vsum_12_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vmul_12_ptr) + $(0), dtype=tl.float32))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(1)])
    tl.store($(candidate_vmul_13_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(1))
    r50_value = (1.0 * tl.load($(candidate_vmul_12_ptr) + $(0), dtype=tl.float32))
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(1)])
    tl.store($(candidate_vmul_14_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(16))
    r51_value = (tl.load($(candidate_vmul_14_ptr) + $(0), dtype=tl.float32) * tl.toReal(tl.load($(candidate_vadd_2_ptr) + r51_z, dtype=tl.int64)))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(16)])
    tl.store($(candidate_vmul_15_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(512))
    r52_value = (tl.load($(candidate_vmul_15_ptr) + (r52_z % $(16)), dtype=tl.float32) * tl.where(tl.load($(candidate_vbitwise_not_1_ptr) + ((r52_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(512)])
    tl.store($(candidate_vmul_16_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(512))
    r53_value = (tl.load($(candidate_vsigmoid_1_ptr) + ((((r53_z / $(16)) % $(32)) * $(16)) + (r53_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_1_ptr) + ((r53_z / $(16)) % $(32)), dtype=tl.float32))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(512)])
    tl.store($(candidate_vdiv_2_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(512))
    r54_value = (tl.load($(candidate_vdiv_2_ptr) + ((((r54_z / $(16)) % $(32)) * $(16)) + (r54_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_1_ptr) + ((r54_z / $(16)) % $(32)), dtype=tl.float32))
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(512)])
    tl.store($(candidate_vdiv_3_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(512))
    r55_value = -tl.load($(candidate_vmul_16_ptr) + ((((r55_z / $(16)) % $(32)) * $(16)) + (r55_z % $(16))), dtype=tl.float32)
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(512)])
    tl.store($(candidate_vneg_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(512))
    r56_value = (tl.load($(candidate_vneg_ptr) + ((((r56_z / $(16)) % $(32)) * $(16)) + (r56_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vdiv_3_ptr) + ((((r56_z / $(16)) % $(32)) * $(16)) + (r56_z % $(16))), dtype=tl.float32))
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(512)])
    tl.store($(candidate_vmul_17_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(512))
    r57_value = (tl.load($(candidate_vmul_16_ptr) + ((((r57_z / $(16)) % $(32)) * $(16)) + (r57_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_1_ptr) + ((r57_z / $(16)) % $(32)), dtype=tl.float32))
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(512)])
    tl.store($(candidate_vdiv_4_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(32))
    r58_lane = tl.arange($(16))
    r58_value = tl.sum(tl.load($(candidate_vmul_17_ptr) + ((r58_z[:, None] * $(16)) + r58_lane[None, :]), dtype=tl.float32), axis=1)
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(32)])
    tl.store($(candidate_vsum_13_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(512))
    r59_value = (tl.load($(candidate_vdiv_4_ptr) + ((((r59_z / $(16)) % $(32)) * $(16)) + (r59_z % $(16))), dtype=tl.float32) + tl.load($(candidate_vsum_13_ptr) + ((r59_z / $(16)) % $(32)), dtype=tl.float32))
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(512)])
    tl.store($(candidate_vadd_3_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(512))
    r60_value = (tl.load($(candidate_vadd_3_ptr) + ((((r60_z / $(16)) % $(32)) * $(16)) + (r60_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsigmoid_1_ptr) + ((((r60_z / $(16)) % $(32)) * $(16)) + (r60_z % $(16))), dtype=tl.float32) * (1.0 - tl.load($(candidate_vsigmoid_1_ptr) + ((((r60_z / $(16)) % $(32)) * $(16)) + (r60_z % $(16))), dtype=tl.float32)))
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(512)])
    tl.store($(candidate_vsigmoid_backward_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(512))
    r61_value = tl.load($(candidate_vsigmoid_backward_ptr) + ((((r61_z / $(16)) % $(32)) * $(16)) + (r61_z % $(16))), dtype=tl.float32)
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(512)])
    tl.store($(candidate_vto_copy_6_ptr) + r61_z, r61_value)
    z = tl.arange($(512))
    value = tl.load($(candidate_vto_copy_6_ptr) + ((((z / $(16)) % $(32)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(512)])
    tl.store($(out_ptr) + z, value)
  }

/-- Finite Real specialization. The non-real NaN literal is assigned the unused
Real payload zero. Its connection to `original` is a separate public goal. -/
def realOriginal : ComputeKernel :=
  let constantf1ebabb642_ptr : RegionName := "constantf1ebabb642"
  let constant0d12534efc_ptr : RegionName := "constant0d12534efc"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan0_ptr : RegionName := "literal_nan0"
  let literal_nan1_ptr : RegionName := "literal_nan1"
  let vge_ptr : RegionName := "vge"
  let vge_1_ptr : RegionName := "vge_1"
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
  let vsum_2_ptr : RegionName := "vsum_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vsigmoid_1_ptr : RegionName := "vsigmoid_1"
  let vsum_7_ptr : RegionName := "vsum_7"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let visnan_1_ptr : RegionName := "visnan_1"
  let vwhere_1_ptr : RegionName := "vwhere_1"
  let vsort_1_value_ptr : RegionName := "vsort_1_value"
  let vsort_1_index_ptr : RegionName := "vsort_1_index"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vscatter_1_ptr : RegionName := "vscatter_1"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vbitwise_not_1_ptr : RegionName := "vbitwise_not_1"
  let vbitwise_and_1_ptr : RegionName := "vbitwise_and_1"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vsum_8_ptr : RegionName := "vsum_8"
  let vsum_9_ptr : RegionName := "vsum_9"
  let vfloor_divide_2_ptr : RegionName := "vfloor_divide_2"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vsum_10_ptr : RegionName := "vsum_10"
  let vfloor_divide_3_ptr : RegionName := "vfloor_divide_3"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vclamp_min_2_ptr : RegionName := "vclamp_min_2"
  let vdiv_4_ptr : RegionName := "vdiv_4"
  let vdiv_5_ptr : RegionName := "vdiv_5"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vclamp_min_3_ptr : RegionName := "vclamp_min_3"
  let vsum_11_ptr : RegionName := "vsum_11"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vsum_12_ptr : RegionName := "vsum_12"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vreciprocal_1_ptr : RegionName := "vreciprocal_1"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vdiv_6_ptr : RegionName := "vdiv_6"
  let vdiv_7_ptr : RegionName := "vdiv_7"
  let vneg_ptr : RegionName := "vneg"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vdiv_8_ptr : RegionName := "vdiv_8"
  let vsum_13_ptr : RegionName := "vsum_13"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  triton {
    r0_z = tl.arange($(1))
    r0_value = 0.0
    tl.store($(literal_nan0_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1))
    r1_value = 0.0
    tl.store($(literal_nan1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32))
    r2_value = (tl.cast((r2_z % $(8)), tl.int64) >= tl.load($(constantf1ebabb642_ptr) + ((r2_z / $(8)) % $(4)), dtype=tl.int64))
    r2_value = tl.broadcast_to(r2_value, [$(32)])
    tl.store($(vge_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(32))
    r3_value = (tl.cast((r3_z % $(8)), tl.int64) >= tl.load($(constant0d12534efc_ptr) + ((r3_z / $(8)) % $(4)), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(32)])
    tl.store($(vge_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(512))
    r4_value = tl.load($(input0_ptr) + ((((r4_z / $(16)) % $(32)) * $(16)) + (r4_z % $(16))), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(512)])
    tl.store($(vto_copy_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r5_z / $(16)) % $(32)) * $(16)) + (r5_z % $(16))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(vsigmoid_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32))
    r6_lane = tl.arange($(16))
    r6_value = tl.sum(tl.load($(vsigmoid_ptr) + ((r6_z[:, None] * $(16)) + r6_lane[None, :]), dtype=tl.float32), axis=1)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(32)])
    tl.store($(vsum_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32))
    r7_value = (tl.load($(vsum_1_ptr) + r7_z, dtype=tl.float32) + 0.00000000000000000001)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32)])
    tl.store($(vadd_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(512))
    r8_value = (tl.load($(vsigmoid_ptr) + ((((r8_z / $(16)) % $(32)) * $(16)) + (r8_z % $(16))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r8_z / $(16)) % $(32)), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(512)])
    tl.store($(vdiv_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(512))
    r9_value = (tl.load($(vdiv_ptr) + ((((r9_z / $(16)) % $(32)) * $(16)) + (r9_z % $(16))), dtype=tl.float32) != tl.load($(vdiv_ptr) + ((((r9_z / $(16)) % $(32)) * $(16)) + (r9_z % $(16))), dtype=tl.float32))
    r9_value = tl.broadcast_to(r9_value, [$(512)])
    tl.store($(visnan_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(512))
    r10_value = tl.where(tl.load($(visnan_ptr) + ((((r10_z / $(16)) % $(32)) * $(16)) + (r10_z % $(16))), dtype=tl.int1), tl.load($(literal_nan0_ptr) + $(0), dtype=tl.float32), tl.load($(vdiv_ptr) + ((((r10_z / $(16)) % $(32)) * $(16)) + (r10_z % $(16))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(512)])
    tl.store($(vwhere_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(512))
    r11_lane = tl.arange($(16))
    r11_row_values = tl.load($(vwhere_ptr) + ((((r11_z[:, None] / $(16)) % $(32)) * $(16)) + (r11_lane[None, :] % $(16))), dtype=tl.float32)
    r11_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r11_other in range($(16)) {
      r11_other_value = tl.load($(vwhere_ptr) + ((((r11_z[:, None] / $(16)) % $(32)) * $(16)) + (r11_other % $(16))), dtype=tl.float32)
      r11_other_nan = r11_other_value != r11_other_value
      r11_row_nan = r11_row_values != r11_row_values
      r11_strict = (r11_other_nan & (~r11_row_nan)) | (((~r11_other_nan) & (~r11_row_nan)) & (r11_other_value > r11_row_values))
      r11_equal = (r11_other_nan & r11_row_nan) | (r11_other_value == r11_row_values)
      r11_rank = r11_rank + tl.where(r11_strict | (r11_equal & (r11_other < r11_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r11_selected = r11_rank == tl.cast((r11_z % $(16))[:, None], tl.int64)
    r11_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r11_cutoff in range($(16)) {
      r11_value = r11_value + tl.cast(tl.sum(r11_selected & (r11_lane[None, :] > r11_cutoff), axis=1), tl.int64)
    }
    r11_value = tl.load($(vwhere_ptr) + ((((r11_z / $(16)) % $(32)) * $(16)) + (tl.cast(r11_value, tl.uint32) % $(16))), dtype=tl.float32)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(512)])
    tl.store($(vsort_value_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(512))
    r12_lane = tl.arange($(16))
    r12_row_values = tl.load($(vwhere_ptr) + ((((r12_z[:, None] / $(16)) % $(32)) * $(16)) + (r12_lane[None, :] % $(16))), dtype=tl.float32)
    r12_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r12_other in range($(16)) {
      r12_other_value = tl.load($(vwhere_ptr) + ((((r12_z[:, None] / $(16)) % $(32)) * $(16)) + (r12_other % $(16))), dtype=tl.float32)
      r12_other_nan = r12_other_value != r12_other_value
      r12_row_nan = r12_row_values != r12_row_values
      r12_strict = (r12_other_nan & (~r12_row_nan)) | (((~r12_other_nan) & (~r12_row_nan)) & (r12_other_value > r12_row_values))
      r12_equal = (r12_other_nan & r12_row_nan) | (r12_other_value == r12_row_values)
      r12_rank = r12_rank + tl.where(r12_strict | (r12_equal & (r12_other < r12_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r12_selected = r12_rank == tl.cast((r12_z % $(16))[:, None], tl.int64)
    r12_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r12_cutoff in range($(16)) {
      r12_value = r12_value + tl.cast(tl.sum(r12_selected & (r12_lane[None, :] > r12_cutoff), axis=1), tl.int64)
    }
    r12_value = tl.broadcast_to(r12_value, [$(512)])
    tl.store($(vsort_index_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = tl.full([], $(0), dtype=tl.int64)
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(vto_copy_1_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(512))
    r14_value = tl.broadcast_to(tl.load($(vto_copy_1_ptr) + ((((r14_z / $(16)) % $(32)) * $(16)) + (r14_z % $(16))), dtype=tl.int64), [$(512)])
    for r14_lane in range($(2)) {
      r14_value = tl.where(tl.load($(vsort_index_ptr) + ((((r14_z / $(16)) % $(32)) * $(16)) + (r14_lane % $(16))), dtype=tl.int64) == tl.cast((r14_z % $(16)), tl.int64), tl.full([], $(1), dtype=tl.int64), r14_value)
    }
    r14_value = tl.broadcast_to(r14_value, [$(512)])
    tl.store($(vscatter_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(512))
    r15_value = (tl.load($(vscatter_ptr) + ((((r15_z / $(16)) % $(32)) * $(16)) + (r15_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r15_value = tl.broadcast_to(r15_value, [$(512)])
    tl.store($(vto_copy_2_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(32))
    r16_value = (~tl.load($(vge_ptr) + ((((r16_z / $(8)) % $(4)) * $(8)) + (r16_z % $(8))), dtype=tl.int1))
    r16_value = tl.broadcast_to(r16_value, [$(32)])
    tl.store($(vbitwise_not_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(512))
    r17_value = (tl.load($(vto_copy_2_ptr) + ((((r17_z / $(16)) % $(32)) * $(16)) + (r17_z % $(16))), dtype=tl.int1) & tl.load($(vbitwise_not_ptr) + ((r17_z / $(16)) % $(32)), dtype=tl.int1))
    r17_value = tl.broadcast_to(r17_value, [$(512)])
    tl.store($(vbitwise_and_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(16))
    r18_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r18_lane in range($(32)) {
      r18_value = r18_value + tl.where(tl.load($(vbitwise_and_ptr) + ((r18_lane * $(16)) + r18_z), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r18_value = tl.broadcast_to(r18_value, [$(16)])
    tl.store($(vsum_2_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(512))
    r19_value = tl.load($(input0_ptr) + ((((r19_z / $(16)) % $(32)) * $(16)) + (r19_z % $(16))), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(512)])
    tl.store($(vto_copy_3_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(512))
    r20_value = tl.sigmoid(tl.load($(vto_copy_3_ptr) + ((((r20_z / $(16)) % $(32)) * $(16)) + (r20_z % $(16))), dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(512)])
    tl.store($(vsigmoid_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(32))
    r21_lane = tl.arange($(16))
    r21_value = tl.sum(tl.load($(vsigmoid_1_ptr) + ((r21_z[:, None] * $(16)) + r21_lane[None, :]), dtype=tl.float32), axis=1)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(32)])
    tl.store($(vsum_7_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(32))
    r22_value = (tl.load($(vsum_7_ptr) + r22_z, dtype=tl.float32) + 0.00000000000000000001)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(32)])
    tl.store($(vadd_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(512))
    r23_value = (tl.load($(vsigmoid_1_ptr) + ((((r23_z / $(16)) % $(32)) * $(16)) + (r23_z % $(16))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r23_z / $(16)) % $(32)), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(512)])
    tl.store($(vdiv_3_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(512))
    r24_value = (tl.load($(vdiv_3_ptr) + ((((r24_z / $(16)) % $(32)) * $(16)) + (r24_z % $(16))), dtype=tl.float32) != tl.load($(vdiv_3_ptr) + ((((r24_z / $(16)) % $(32)) * $(16)) + (r24_z % $(16))), dtype=tl.float32))
    r24_value = tl.broadcast_to(r24_value, [$(512)])
    tl.store($(visnan_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(512))
    r25_value = tl.where(tl.load($(visnan_1_ptr) + ((((r25_z / $(16)) % $(32)) * $(16)) + (r25_z % $(16))), dtype=tl.int1), tl.load($(literal_nan1_ptr) + $(0), dtype=tl.float32), tl.load($(vdiv_3_ptr) + ((((r25_z / $(16)) % $(32)) * $(16)) + (r25_z % $(16))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(512)])
    tl.store($(vwhere_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(512))
    r26_lane = tl.arange($(16))
    r26_row_values = tl.load($(vwhere_1_ptr) + ((((r26_z[:, None] / $(16)) % $(32)) * $(16)) + (r26_lane[None, :] % $(16))), dtype=tl.float32)
    r26_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r26_other in range($(16)) {
      r26_other_value = tl.load($(vwhere_1_ptr) + ((((r26_z[:, None] / $(16)) % $(32)) * $(16)) + (r26_other % $(16))), dtype=tl.float32)
      r26_other_nan = r26_other_value != r26_other_value
      r26_row_nan = r26_row_values != r26_row_values
      r26_strict = (r26_other_nan & (~r26_row_nan)) | (((~r26_other_nan) & (~r26_row_nan)) & (r26_other_value > r26_row_values))
      r26_equal = (r26_other_nan & r26_row_nan) | (r26_other_value == r26_row_values)
      r26_rank = r26_rank + tl.where(r26_strict | (r26_equal & (r26_other < r26_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r26_selected = r26_rank == tl.cast((r26_z % $(16))[:, None], tl.int64)
    r26_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r26_cutoff in range($(16)) {
      r26_value = r26_value + tl.cast(tl.sum(r26_selected & (r26_lane[None, :] > r26_cutoff), axis=1), tl.int64)
    }
    r26_value = tl.load($(vwhere_1_ptr) + ((((r26_z / $(16)) % $(32)) * $(16)) + (tl.cast(r26_value, tl.uint32) % $(16))), dtype=tl.float32)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(512)])
    tl.store($(vsort_1_value_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(512))
    r27_lane = tl.arange($(16))
    r27_row_values = tl.load($(vwhere_1_ptr) + ((((r27_z[:, None] / $(16)) % $(32)) * $(16)) + (r27_lane[None, :] % $(16))), dtype=tl.float32)
    r27_rank = tl.full([$(512), $(16)], $(0), dtype=tl.int64)
    for r27_other in range($(16)) {
      r27_other_value = tl.load($(vwhere_1_ptr) + ((((r27_z[:, None] / $(16)) % $(32)) * $(16)) + (r27_other % $(16))), dtype=tl.float32)
      r27_other_nan = r27_other_value != r27_other_value
      r27_row_nan = r27_row_values != r27_row_values
      r27_strict = (r27_other_nan & (~r27_row_nan)) | (((~r27_other_nan) & (~r27_row_nan)) & (r27_other_value > r27_row_values))
      r27_equal = (r27_other_nan & r27_row_nan) | (r27_other_value == r27_row_values)
      r27_rank = r27_rank + tl.where(r27_strict | (r27_equal & (r27_other < r27_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r27_selected = r27_rank == tl.cast((r27_z % $(16))[:, None], tl.int64)
    r27_value = tl.full([$(512)], $(0), dtype=tl.int64)
    for r27_cutoff in range($(16)) {
      r27_value = r27_value + tl.cast(tl.sum(r27_selected & (r27_lane[None, :] > r27_cutoff), axis=1), tl.int64)
    }
    r27_value = tl.broadcast_to(r27_value, [$(512)])
    tl.store($(vsort_1_index_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(512))
    r28_value = tl.full([], $(0), dtype=tl.int64)
    r28_value = tl.broadcast_to(r28_value, [$(512)])
    tl.store($(vto_copy_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(512))
    r29_value = tl.broadcast_to(tl.load($(vto_copy_4_ptr) + ((((r29_z / $(16)) % $(32)) * $(16)) + (r29_z % $(16))), dtype=tl.int64), [$(512)])
    for r29_lane in range($(2)) {
      r29_value = tl.where(tl.load($(vsort_1_index_ptr) + ((((r29_z / $(16)) % $(32)) * $(16)) + (r29_lane % $(16))), dtype=tl.int64) == tl.cast((r29_z % $(16)), tl.int64), tl.full([], $(1), dtype=tl.int64), r29_value)
    }
    r29_value = tl.broadcast_to(r29_value, [$(512)])
    tl.store($(vscatter_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(512))
    r30_value = (tl.load($(vscatter_1_ptr) + ((((r30_z / $(16)) % $(32)) * $(16)) + (r30_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r30_value = tl.broadcast_to(r30_value, [$(512)])
    tl.store($(vto_copy_5_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(32))
    r31_value = (~tl.load($(vge_1_ptr) + ((((r31_z / $(8)) % $(4)) * $(8)) + (r31_z % $(8))), dtype=tl.int1))
    r31_value = tl.broadcast_to(r31_value, [$(32)])
    tl.store($(vbitwise_not_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(512))
    r32_value = (tl.load($(vto_copy_5_ptr) + ((((r32_z / $(16)) % $(32)) * $(16)) + (r32_z % $(16))), dtype=tl.int1) & tl.load($(vbitwise_not_1_ptr) + ((r32_z / $(16)) % $(32)), dtype=tl.int1))
    r32_value = tl.broadcast_to(r32_value, [$(512)])
    tl.store($(vbitwise_and_1_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(512))
    r33_value = (tl.load($(vdiv_3_ptr) + ((((r33_z / $(16)) % $(32)) * $(16)) + (r33_z % $(16))), dtype=tl.float32) * tl.where(tl.load($(vbitwise_not_1_ptr) + ((r33_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(512)])
    tl.store($(vmul_8_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(16))
    r34_value = tl.full([$(16)], $(0), dtype=tl.int64)
    for r34_lane in range($(32)) {
      r34_value = r34_value + tl.where(tl.load($(vbitwise_and_1_ptr) + ((r34_lane * $(16)) + r34_z), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r34_value = tl.broadcast_to(r34_value, [$(16)])
    tl.store($(vsum_8_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(1))
    r35_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r35_lane in range($(16)) {
      r35_value = r35_value + tl.load($(vsum_8_ptr) + r35_lane, dtype=tl.int64)
    }
    r35_value = tl.broadcast_to(r35_value, [$(1)])
    tl.store($(vsum_9_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(1))
    r36_value = (tl.load($(vsum_9_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r36_value = tl.broadcast_to(r36_value, [$(1)])
    tl.store($(vfloor_divide_2_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(16))
    r37_value = (tl.load($(vsum_2_ptr) + r37_z, dtype=tl.int64) + tl.load($(vsum_8_ptr) + r37_z, dtype=tl.int64))
    r37_value = tl.broadcast_to(r37_value, [$(16)])
    tl.store($(vadd_2_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(1))
    r38_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r38_lane in range($(16)) {
      r38_value = r38_value + tl.load($(vadd_2_ptr) + r38_lane, dtype=tl.int64)
    }
    r38_value = tl.broadcast_to(r38_value, [$(1)])
    tl.store($(vsum_10_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(1))
    r39_value = (tl.load($(vsum_10_ptr) + $(0), dtype=tl.int64) / tl.full([], $(2), dtype=tl.int64))
    r39_value = tl.broadcast_to(r39_value, [$(1)])
    tl.store($(vfloor_divide_3_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(1))
    r40_value = (tl.full([], $(2), dtype=tl.int64) * tl.load($(vfloor_divide_2_ptr) + $(0), dtype=tl.int64))
    r40_value = tl.broadcast_to(r40_value, [$(1)])
    tl.store($(vmul_9_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(1))
    r41_value = tl.maximum(tl.load($(vfloor_divide_3_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r41_value = tl.broadcast_to(r41_value, [$(1)])
    tl.store($(vclamp_min_2_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(1))
    r42_value = (tl.toReal(tl.load($(vmul_9_ptr) + $(0), dtype=tl.int64)) / tl.toReal(tl.load($(vclamp_min_2_ptr) + $(0), dtype=tl.int64)))
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(1)])
    tl.store($(vdiv_4_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(16))
    r43_value = (tl.toReal(tl.load($(vadd_2_ptr) + r43_z, dtype=tl.int64)) / tl.toReal(tl.full([], $(2), dtype=tl.int64)))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(16)])
    tl.store($(vdiv_5_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(16))
    r44_value = (tl.load($(vdiv_5_ptr) + r44_z, dtype=tl.float32) * tl.load($(vdiv_4_ptr) + $(0), dtype=tl.float32))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(16)])
    tl.store($(vmul_10_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(1))
    r45_value = tl.maximum(tl.load($(vfloor_divide_2_ptr) + $(0), dtype=tl.int64), tl.full([], $(1), dtype=tl.int64))
    r45_value = tl.broadcast_to(r45_value, [$(1)])
    tl.store($(vclamp_min_3_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(16))
    r46_lane = tl.arange($(32))
    r46_value = tl.sum(tl.load($(vmul_8_ptr) + ((r46_lane[None, :] * $(16)) + r46_z[:, None]), dtype=tl.float32), axis=1)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(16)])
    tl.store($(vsum_11_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(16))
    r47_value = (tl.load($(vsum_11_ptr) + r47_z, dtype=tl.float32) * tl.load($(vmul_10_ptr) + r47_z, dtype=tl.float32))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(16)])
    tl.store($(vmul_11_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(1))
    r48_lane = tl.arange($(16))
    r48_value = tl.sum(tl.load($(vmul_11_ptr) + r48_lane[None, :], dtype=tl.float32), axis=1)
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(1)])
    tl.store($(vsum_12_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(1))
    r49_value = (tl.load($(vclamp_min_3_ptr) + $(0), dtype=tl.int64) * tl.full([], $(2), dtype=tl.int64))
    r49_value = tl.broadcast_to(r49_value, [$(1)])
    tl.store($(vmul_12_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(1))
    r50_value = (tl.load($(vmul_12_ptr) + $(0), dtype=tl.int64) * tl.load($(vclamp_min_3_ptr) + $(0), dtype=tl.int64))
    r50_value = tl.broadcast_to(r50_value, [$(1)])
    tl.store($(vmul_13_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(1))
    r51_value = 1.0 / tl.toReal(tl.load($(vmul_13_ptr) + $(0), dtype=tl.int64))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(1)])
    tl.store($(vreciprocal_1_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(1))
    r52_value = (tl.load($(vreciprocal_1_ptr) + $(0), dtype=tl.float32) * 2.08)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(1)])
    tl.store($(vmul_14_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(1))
    r53_value = (tl.load($(vsum_12_ptr) + $(0), dtype=tl.float32) * tl.load($(vmul_14_ptr) + $(0), dtype=tl.float32))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(1)])
    tl.store($(vmul_15_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(1))
    r54_value = (1.0 * tl.load($(vmul_14_ptr) + $(0), dtype=tl.float32))
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(1)])
    tl.store($(vmul_16_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(16))
    r55_value = (tl.load($(vmul_16_ptr) + $(0), dtype=tl.float32) * tl.load($(vmul_10_ptr) + r55_z, dtype=tl.float32))
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(16)])
    tl.store($(vmul_17_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(512))
    r56_value = (tl.load($(vmul_17_ptr) + (r56_z % $(16)), dtype=tl.float32) * tl.where(tl.load($(vbitwise_not_1_ptr) + ((r56_z / $(16)) % $(32)), dtype=tl.int1), 1.0, 0.0))
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(512)])
    tl.store($(vmul_18_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(512))
    r57_value = (tl.load($(vsigmoid_1_ptr) + ((((r57_z / $(16)) % $(32)) * $(16)) + (r57_z % $(16))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r57_z / $(16)) % $(32)), dtype=tl.float32))
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(512)])
    tl.store($(vdiv_6_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(512))
    r58_value = (tl.load($(vdiv_6_ptr) + ((((r58_z / $(16)) % $(32)) * $(16)) + (r58_z % $(16))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r58_z / $(16)) % $(32)), dtype=tl.float32))
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(512)])
    tl.store($(vdiv_7_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(512))
    r59_value = -tl.load($(vmul_18_ptr) + ((((r59_z / $(16)) % $(32)) * $(16)) + (r59_z % $(16))), dtype=tl.float32)
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(512)])
    tl.store($(vneg_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(512))
    r60_value = (tl.load($(vneg_ptr) + ((((r60_z / $(16)) % $(32)) * $(16)) + (r60_z % $(16))), dtype=tl.float32) * tl.load($(vdiv_7_ptr) + ((((r60_z / $(16)) % $(32)) * $(16)) + (r60_z % $(16))), dtype=tl.float32))
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(512)])
    tl.store($(vmul_19_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(512))
    r61_value = (tl.load($(vmul_18_ptr) + ((((r61_z / $(16)) % $(32)) * $(16)) + (r61_z % $(16))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r61_z / $(16)) % $(32)), dtype=tl.float32))
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(512)])
    tl.store($(vdiv_8_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(32))
    r62_lane = tl.arange($(16))
    r62_value = tl.sum(tl.load($(vmul_19_ptr) + ((r62_z[:, None] * $(16)) + r62_lane[None, :]), dtype=tl.float32), axis=1)
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(32)])
    tl.store($(vsum_13_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(512))
    r63_value = (tl.load($(vdiv_8_ptr) + ((((r63_z / $(16)) % $(32)) * $(16)) + (r63_z % $(16))), dtype=tl.float32) + tl.load($(vsum_13_ptr) + ((r63_z / $(16)) % $(32)), dtype=tl.float32))
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(512)])
    tl.store($(vadd_3_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(512))
    r64_value = (tl.load($(vadd_3_ptr) + ((((r64_z / $(16)) % $(32)) * $(16)) + (r64_z % $(16))), dtype=tl.float32) * tl.load($(vsigmoid_1_ptr) + ((((r64_z / $(16)) % $(32)) * $(16)) + (r64_z % $(16))), dtype=tl.float32) * (1.0 - tl.load($(vsigmoid_1_ptr) + ((((r64_z / $(16)) % $(32)) * $(16)) + (r64_z % $(16))), dtype=tl.float32)))
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(512)])
    tl.store($(vsigmoid_backward_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(512))
    r65_value = tl.load($(vsigmoid_backward_ptr) + ((((r65_z / $(16)) % $(32)) * $(16)) + (r65_z % $(16))), dtype=tl.float32)
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(512)])
    tl.store($(vto_copy_6_ptr) + r65_z, r65_value)
    z = tl.arange($(512))
    value = tl.load($(vto_copy_6_ptr) + ((((z / $(16)) % $(32)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(512)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_masked_moe_global_aux_dx
