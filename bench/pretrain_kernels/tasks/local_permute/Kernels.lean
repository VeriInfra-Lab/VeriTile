import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_permute
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
  let candidate_vsort_value_ptr : RegionName := "candidate_vsort_value"
  let candidate_vsort_index_ptr : RegionName := "candidate_vsort_index"
  let candidate_vremainder_ptr : RegionName := "candidate_vremainder"
  let candidate_vindex_select_ptr : RegionName := "candidate_vindex_select"
  let candidate_vscatter_add_ptr : RegionName := "candidate_vscatter_add"
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
    r4_lane = tl.arange($(32768))
    r4_row_values = tl.load($(candidate_vscatter_ptr) + (((r4_lane[None, :] % $(256)) * $(128)) + ((r4_lane[None, :] / $(256)) % $(128))), dtype=tl.int1)
    r4_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r4_other in range($(32768)) {
      r4_rank = r4_rank + tl.where(((tl.load($(candidate_vscatter_ptr) + (((r4_other % $(256)) * $(128)) + ((r4_other / $(256)) % $(128))), dtype=tl.int1) & (~r4_row_values)) | (((tl.load($(candidate_vscatter_ptr) + (((r4_other % $(256)) * $(128)) + ((r4_other / $(256)) % $(128))), dtype=tl.int1) & r4_row_values) | ((~tl.load($(candidate_vscatter_ptr) + (((r4_other % $(256)) * $(128)) + ((r4_other / $(256)) % $(128))), dtype=tl.int1)) & (~r4_row_values))) & (r4_other < r4_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r4_selected = r4_rank == tl.cast(r4_z[:, None], tl.int64)
    r4_value = tl.sum(r4_selected & r4_row_values, axis=1) > $(0)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(candidate_vsort_value_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_lane = tl.arange($(32768))
    r5_row_values = tl.load($(candidate_vscatter_ptr) + (((r5_lane[None, :] % $(256)) * $(128)) + ((r5_lane[None, :] / $(256)) % $(128))), dtype=tl.int1)
    r5_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r5_other in range($(32768)) {
      r5_rank = r5_rank + tl.where(((tl.load($(candidate_vscatter_ptr) + (((r5_other % $(256)) * $(128)) + ((r5_other / $(256)) % $(128))), dtype=tl.int1) & (~r5_row_values)) | (((tl.load($(candidate_vscatter_ptr) + (((r5_other % $(256)) * $(128)) + ((r5_other / $(256)) % $(128))), dtype=tl.int1) & r5_row_values) | ((~tl.load($(candidate_vscatter_ptr) + (((r5_other % $(256)) * $(128)) + ((r5_other / $(256)) % $(128))), dtype=tl.int1)) & (~r5_row_values))) & (r5_other < r5_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r5_selected = r5_rank == tl.cast(r5_z[:, None], tl.int64)
    r5_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r5_cutoff in range($(32768)) {
      r5_value = r5_value + tl.cast(tl.sum(r5_selected & (r5_lane[None, :] > r5_cutoff), axis=1), tl.int64)
    }
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(candidate_vsort_index_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(256))
    r6_value = (tl.load($(candidate_vsort_index_ptr) + r6_z, dtype=tl.int64) % tl.full([], $(256), dtype=tl.int64))
    r6_value = tl.broadcast_to(r6_value, [$(256)])
    tl.store($(candidate_vremainder_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(524288))
    r7_value = tl.load($(candidate_vto_copy_1_ptr) + (((tl.load($(candidate_vremainder_ptr) + ((r7_z / $(2048)) % $(256)), dtype=tl.int64) % $(256)) * $(2048)) + (r7_z % $(2048))), dtype=tl.bfloat16)
    r7_value = r7_value.to(tl.bfloat16)
    r7_value = tl.broadcast_to(r7_value, [$(524288)])
    tl.store($(candidate_vindex_select_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(524288))
    r8_value = tl.broadcast_to(tl.cast(0.0, tl.bfloat16), [$(524288)])
    for r8_lane in range($(256)) {
      r8_value = tl.where(tl.load($(candidate_vremainder_ptr) + r8_lane, dtype=tl.int64) == tl.cast(((r8_z / $(2048)) % $(256)), tl.int64), (r8_value + tl.load($(candidate_vindex_select_ptr) + (((r8_lane % $(256)) * $(2048)) + (r8_z % $(2048))), dtype=tl.bfloat16)), r8_value)
    }
    r8_value = r8_value.to(tl.bfloat16)
    r8_value = tl.broadcast_to(r8_value, [$(524288)])
    tl.store($(candidate_vscatter_add_ptr) + r8_z, r8_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vscatter_add_ptr) + ((((z / $(2048)) % $(256)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_permute
