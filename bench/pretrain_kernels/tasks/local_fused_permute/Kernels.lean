import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_permute
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  triton {
    r0_z = tl.arange($(524288))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(2048)) % $(256)) * $(2048)) + (r0_z % $(2048))), dtype=tl.float64)
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(524288)])
    tl.store($(vto_copy_1_ptr) + r0_z, r0_value)
    z = tl.arange($(524288))
    value = tl.load($(vto_copy_1_ptr) + ((((z / $(2048)) % $(256)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let parameter0_ptr : RegionName := "parameter0"
  let out_ptr : RegionName := "out"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let candidate_vtopk_value_ptr : RegionName := "candidate_vtopk_value"
  let candidate_vtopk_index_ptr : RegionName := "candidate_vtopk_index"
  let candidate_vscatter_ptr : RegionName := "candidate_vscatter"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vcumsum_ptr : RegionName := "candidate_vcumsum"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vcumsum_1_ptr : RegionName := "candidate_vcumsum_1"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vwhere_1_ptr : RegionName := "candidate_vwhere_1"
  let candidate_vwhere_2_ptr : RegionName := "candidate_vwhere_2"
  let candidate_vscatter_add_ptr : RegionName := "candidate_vscatter_add"
  let candidate_vge_ptr : RegionName := "candidate_vge"
  let candidate_vclamp_min_ptr : RegionName := "candidate_vclamp_min"
  let candidate_vindex_ptr : RegionName := "candidate_vindex"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vwhere_3_ptr : RegionName := "candidate_vwhere_3"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vto_copy_7_ptr : RegionName := "candidate_vto_copy_7"
  triton {
    r0_z = tl.arange($(256))
    r0_lane = tl.arange($(128))
    r0_row_values = tl.load($(parameter0_ptr) + ((r0_z[:, None] * $(128)) + (r0_lane[None, :] % $(128))), dtype=tl.float32)
    r0_rank = tl.full([$(256), $(128)], $(0), dtype=tl.int64)
    for r0_other in range($(128)) {
      r0_other_value = tl.load($(parameter0_ptr) + ((r0_z[:, None] * $(128)) + (r0_other % $(128))), dtype=tl.float32)
      r0_other_nan = r0_other_value != r0_other_value
      r0_row_nan = r0_row_values != r0_row_values
      r0_strict = (r0_other_nan & (~r0_row_nan)) | (((~r0_other_nan) & (~r0_row_nan)) & (r0_other_value > r0_row_values))
      r0_equal = (r0_other_nan & r0_row_nan) | (r0_other_value == r0_row_values)
      r0_rank = r0_rank + tl.where(r0_strict | (r0_equal & (r0_other < r0_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r0_selected = r0_rank == tl.cast($(0)[:, None], tl.int64)
    r0_value = tl.full([$(256)], $(0), dtype=tl.int64)
    for r0_cutoff in range($(128)) {
      r0_value = r0_value + tl.cast(tl.sum(r0_selected & (r0_lane[None, :] > r0_cutoff), axis=1), tl.int64)
    }
    r0_value = tl.load($(parameter0_ptr) + ((r0_z * $(128)) + (tl.cast(r0_value, tl.uint32) % $(128))), dtype=tl.float32)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(256)])
    tl.store($(candidate_vtopk_value_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_lane = tl.arange($(128))
    r1_row_values = tl.load($(parameter0_ptr) + ((r1_z[:, None] * $(128)) + (r1_lane[None, :] % $(128))), dtype=tl.float32)
    r1_rank = tl.full([$(256), $(128)], $(0), dtype=tl.int64)
    for r1_other in range($(128)) {
      r1_other_value = tl.load($(parameter0_ptr) + ((r1_z[:, None] * $(128)) + (r1_other % $(128))), dtype=tl.float32)
      r1_other_nan = r1_other_value != r1_other_value
      r1_row_nan = r1_row_values != r1_row_values
      r1_strict = (r1_other_nan & (~r1_row_nan)) | (((~r1_other_nan) & (~r1_row_nan)) & (r1_other_value > r1_row_values))
      r1_equal = (r1_other_nan & r1_row_nan) | (r1_other_value == r1_row_values)
      r1_rank = r1_rank + tl.where(r1_strict | (r1_equal & (r1_other < r1_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r1_selected = r1_rank == tl.cast($(0)[:, None], tl.int64)
    r1_value = tl.full([$(256)], $(0), dtype=tl.int64)
    for r1_cutoff in range($(128)) {
      r1_value = r1_value + tl.cast(tl.sum(r1_selected & (r1_lane[None, :] > r1_cutoff), axis=1), tl.int64)
    }
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(candidate_vtopk_index_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32768))
    r2_value = tl.broadcast_to(false, [$(32768)])
    for r2_lane in range($(1)) {
      r2_value = tl.where(tl.load($(candidate_vtopk_index_ptr) + ((r2_z / $(128)) % $(256)), dtype=tl.int64) == tl.cast((r2_z % $(128)), tl.int64), true, r2_value)
    }
    r2_value = tl.broadcast_to(r2_value, [$(32768)])
    tl.store($(candidate_vscatter_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(524288))
    r3_value = tl.load($(input0_ptr) + ((((r3_z / $(2048)) % $(256)) * $(2048)) + (r3_z % $(2048))), dtype=tl.float64)
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(524288)])
    tl.store($(candidate_vto_copy_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = tl.where(tl.load($(candidate_vscatter_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(candidate_vto_copy_2_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (r5_z % $(128))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(candidate_vto_copy_3_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32768))
    r6_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r6_z / $(128)) % $(256)) * $(128)) + (r6_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(candidate_vto_copy_4_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(128))
    r7_value = tl.full([$(128)], $(0), dtype=tl.int64)
    for r7_lane in range($(256)) {
      r7_value = r7_value + tl.load($(candidate_vto_copy_4_ptr) + ((r7_lane * $(128)) + r7_z), dtype=tl.int64)
    }
    r7_value = tl.broadcast_to(r7_value, [$(128)])
    tl.store($(candidate_vsum_2_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(128))
    r8_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(128)])
    for r8_lane in range($(128)) {
      r8_value = tl.where(r8_lane <= r8_z, (r8_value + tl.load($(candidate_vsum_2_ptr) + (r8_lane % $(128)), dtype=tl.int64)), r8_value)
    }
    r8_value = tl.broadcast_to(r8_value, [$(128)])
    tl.store($(candidate_vcumsum_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(128))
    r9_value = (tl.load($(candidate_vcumsum_ptr) + r9_z, dtype=tl.int64) - tl.load($(candidate_vsum_2_ptr) + r9_z, dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(128)])
    tl.store($(candidate_vsub_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(32768))
    r10_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r10_z / $(128)) % $(256)) * $(128)) + (r10_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(candidate_vto_copy_5_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(32768))
    r11_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(32768)])
    for r11_lane in range($(256)) {
      r11_value = tl.where(r11_lane <= ((r11_z / $(128)) % $(256)), (r11_value + tl.load($(candidate_vto_copy_5_ptr) + (((r11_lane % $(256)) * $(128)) + (r11_z % $(128))), dtype=tl.int64)), r11_value)
    }
    r11_value = tl.broadcast_to(r11_value, [$(32768)])
    tl.store($(candidate_vcumsum_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(32768))
    r12_value = (tl.load($(candidate_vcumsum_1_ptr) + ((((r12_z / $(128)) % $(256)) * $(128)) + (r12_z % $(128))), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r12_value = tl.broadcast_to(r12_value, [$(32768)])
    tl.store($(candidate_vsub_1_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(32768))
    r13_value = (tl.load($(candidate_vsub_1_ptr) + ((((r13_z / $(128)) % $(256)) * $(128)) + (r13_z % $(128))), dtype=tl.int64) + tl.load($(candidate_vsub_ptr) + (r13_z % $(128)), dtype=tl.int64))
    r13_value = tl.broadcast_to(r13_value, [$(32768)])
    tl.store($(candidate_vadd_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(32768))
    r14_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r14_z / $(128)) % $(256)) * $(128)) + (r14_z % $(128))), dtype=tl.int1), tl.load($(candidate_vadd_ptr) + ((((r14_z / $(128)) % $(256)) * $(128)) + (r14_z % $(128))), dtype=tl.int64), (-tl.full([], $(1), dtype=tl.int64)))
    r14_value = tl.broadcast_to(r14_value, [$(32768)])
    tl.store($(candidate_vwhere_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(32768))
    r15_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r15_z / $(128)) % $(256)) * $(128)) + (r15_z % $(128))), dtype=tl.int1), tl.load($(candidate_vwhere_ptr) + ((((r15_z / $(128)) % $(256)) * $(128)) + (r15_z % $(128))), dtype=tl.int64), tl.full([], $(256), dtype=tl.int64))
    r15_value = tl.broadcast_to(r15_value, [$(32768)])
    tl.store($(candidate_vwhere_1_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(67108864))
    r16_value = tl.where(tl.load($(candidate_vto_copy_3_ptr) + ((((r16_z / $(262144)) % $(256)) * $(128)) + ((r16_z / $(2048)) % $(128))), dtype=tl.int1), tl.load($(candidate_vto_copy_1_ptr) + ((((r16_z / $(262144)) % $(256)) * $(2048)) + (r16_z % $(2048))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r16_value = r16_value.to(tl.bfloat16)
    r16_value = tl.broadcast_to(r16_value, [$(67108864)])
    tl.store($(candidate_vwhere_2_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(526336))
    r17_value = tl.broadcast_to(tl.cast(0.0, tl.bfloat16), [$(526336)])
    for r17_lane in range($(32768)) {
      r17_value = tl.where(tl.load($(candidate_vwhere_1_ptr) + ((((r17_lane / $(128)) % $(256)) * $(128)) + (r17_lane % $(128))), dtype=tl.int64) == tl.cast(((r17_z / $(2048)) % $(257)), tl.int64), (r17_value + tl.load($(candidate_vwhere_2_ptr) + ((((((r17_lane * $(2048)) + (r17_z % $(2048))) / $(262144)) % $(256)) * $(262144)) + ((r17_lane % $(128)) * $(2048)) + (r17_z % $(2048))), dtype=tl.bfloat16)), r17_value)
    }
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(526336)])
    tl.store($(candidate_vscatter_add_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = (tl.load($(candidate_vwhere_ptr) + ((((r18_z / $(128)) % $(256)) * $(128)) + (r18_z % $(128))), dtype=tl.int64) >= tl.full([], $(0), dtype=tl.int64))
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(candidate_vge_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32768))
    r19_value = tl.maximum(tl.load($(candidate_vwhere_ptr) + ((((r19_z / $(128)) % $(256)) * $(128)) + (r19_z % $(128))), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r19_value = tl.broadcast_to(r19_value, [$(32768)])
    tl.store($(candidate_vclamp_min_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(67108864))
    r20_value = tl.load($(candidate_vscatter_add_ptr) + (((tl.load($(candidate_vclamp_min_ptr) + ((((((r20_z / $(2048)) % $(32768)) / $(128)) % $(256)) * $(128)) + (((r20_z / $(2048)) % $(32768)) % $(128))), dtype=tl.int64) % $(257)) * $(2048)) + (r20_z % $(2048))), dtype=tl.bfloat16)
    r20_value = r20_value.to(tl.bfloat16)
    r20_value = tl.broadcast_to(r20_value, [$(67108864)])
    tl.store($(candidate_vindex_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(67108864))
    r21_value = tl.load($(candidate_vindex_ptr) + ((((r21_z / $(262144)) % $(256)) * $(262144)) + (((r21_z / $(2048)) % $(128)) * $(2048)) + (r21_z % $(2048))), dtype=tl.bfloat16)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(67108864)])
    tl.store($(candidate_vto_copy_6_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(67108864))
    r22_value = tl.where(tl.load($(candidate_vge_ptr) + ((((r22_z / $(262144)) % $(256)) * $(128)) + ((r22_z / $(2048)) % $(128))), dtype=tl.int1), tl.load($(candidate_vto_copy_6_ptr) + ((((r22_z / $(262144)) % $(256)) * $(262144)) + (((r22_z / $(2048)) % $(128)) * $(2048)) + (r22_z % $(2048))), dtype=tl.float32), 0.0)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(67108864)])
    tl.store($(candidate_vwhere_3_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(524288))
    r23_lane = tl.arange($(128))
    r23_value = tl.sum(tl.load($(candidate_vwhere_3_ptr) + ((((r23_z[:, None] / $(2048)) % $(256)) * $(262144)) + (r23_lane[None, :] * $(2048)) + (r23_z[:, None] % $(2048))), dtype=tl.float32), axis=1)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(524288)])
    tl.store($(candidate_vsum_3_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(524288))
    r24_value = tl.load($(candidate_vsum_3_ptr) + ((((r24_z / $(2048)) % $(256)) * $(2048)) + (r24_z % $(2048))), dtype=tl.float32)
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(524288)])
    tl.store($(candidate_vto_copy_7_ptr) + r24_z, r24_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vto_copy_7_ptr) + ((((z / $(2048)) % $(256)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_permute
