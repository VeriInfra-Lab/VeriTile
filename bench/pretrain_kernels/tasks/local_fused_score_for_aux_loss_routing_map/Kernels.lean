import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss_routing_map
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan_ptr : RegionName := "literal_nan"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_ptr : RegionName := "vadd"
  let vdiv_ptr : RegionName := "vdiv"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vscatter_ptr : RegionName := "vscatter"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(32768))
    r1_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r1_z / $(128)) % $(256)) * $(128)) + (r1_z % $(128))), dtype=tl.float32))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(32768)])
    tl.store($(vsigmoid_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_lane = tl.arange($(128))
    r2_value = tl.sum(tl.load($(vsigmoid_ptr) + ((r2_z[:, None] * $(128)) + r2_lane[None, :]), dtype=tl.float32), axis=1)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(vsum_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(256))
    r3_value = (tl.load($(vsum_1_ptr) + r3_z, dtype=tl.float32) + 0.00000000000000000001)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(256)])
    tl.store($(vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = (tl.load($(vsigmoid_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r4_z / $(128)) % $(256)), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(vdiv_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_value = (tl.load($(vdiv_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (r5_z % $(128))), dtype=tl.float32) != tl.load($(vdiv_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (r5_z % $(128))), dtype=tl.float32))
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(visnan_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32768))
    r6_value = tl.where(tl.load($(visnan_ptr) + ((((r6_z / $(128)) % $(256)) * $(128)) + (r6_z % $(128))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(vdiv_ptr) + ((((r6_z / $(128)) % $(256)) * $(128)) + (r6_z % $(128))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(vwhere_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_lane = tl.arange($(128))
    r7_row_values = tl.load($(vwhere_ptr) + ((((r7_z[:, None] / $(128)) % $(256)) * $(128)) + (r7_lane[None, :] % $(128))), dtype=tl.float32)
    r7_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r7_other in range($(128)) {
      r7_other_value = tl.load($(vwhere_ptr) + ((((r7_z[:, None] / $(128)) % $(256)) * $(128)) + (r7_other % $(128))), dtype=tl.float32)
      r7_other_nan = r7_other_value != r7_other_value
      r7_row_nan = r7_row_values != r7_row_values
      r7_strict = (r7_other_nan & (~r7_row_nan)) | (((~r7_other_nan) & (~r7_row_nan)) & (r7_other_value > r7_row_values))
      r7_equal = (r7_other_nan & r7_row_nan) | (r7_other_value == r7_row_values)
      r7_rank = r7_rank + tl.where(r7_strict | (r7_equal & (r7_other < r7_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r7_selected = r7_rank == tl.cast((r7_z % $(128))[:, None], tl.int64)
    r7_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r7_cutoff in range($(128)) {
      r7_value = r7_value + tl.cast(tl.sum(r7_selected & (r7_lane[None, :] > r7_cutoff), axis=1), tl.int64)
    }
    r7_value = tl.load($(vwhere_ptr) + ((((r7_z / $(128)) % $(256)) * $(128)) + (tl.cast(r7_value, tl.uint32) % $(128))), dtype=tl.float32)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32768)])
    tl.store($(vsort_value_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32768))
    r8_lane = tl.arange($(128))
    r8_row_values = tl.load($(vwhere_ptr) + ((((r8_z[:, None] / $(128)) % $(256)) * $(128)) + (r8_lane[None, :] % $(128))), dtype=tl.float32)
    r8_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r8_other in range($(128)) {
      r8_other_value = tl.load($(vwhere_ptr) + ((((r8_z[:, None] / $(128)) % $(256)) * $(128)) + (r8_other % $(128))), dtype=tl.float32)
      r8_other_nan = r8_other_value != r8_other_value
      r8_row_nan = r8_row_values != r8_row_values
      r8_strict = (r8_other_nan & (~r8_row_nan)) | (((~r8_other_nan) & (~r8_row_nan)) & (r8_other_value > r8_row_values))
      r8_equal = (r8_other_nan & r8_row_nan) | (r8_other_value == r8_row_values)
      r8_rank = r8_rank + tl.where(r8_strict | (r8_equal & (r8_other < r8_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r8_selected = r8_rank == tl.cast((r8_z % $(128))[:, None], tl.int64)
    r8_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r8_cutoff in range($(128)) {
      r8_value = r8_value + tl.cast(tl.sum(r8_selected & (r8_lane[None, :] > r8_cutoff), axis=1), tl.int64)
    }
    r8_value = tl.broadcast_to(r8_value, [$(32768)])
    tl.store($(vsort_index_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(32768))
    r9_value = tl.full([], $(0), dtype=tl.int64)
    r9_value = tl.broadcast_to(r9_value, [$(32768)])
    tl.store($(vto_copy_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(32768))
    r10_value = tl.broadcast_to(tl.load($(vto_copy_ptr) + ((((r10_z / $(128)) % $(256)) * $(128)) + (r10_z % $(128))), dtype=tl.int64), [$(32768)])
    for r10_lane in range($(16)) {
      r10_value = tl.where(tl.load($(vsort_index_ptr) + ((((r10_z / $(128)) % $(256)) * $(128)) + (r10_lane % $(128))), dtype=tl.int64) == tl.cast((r10_z % $(128)), tl.int64), tl.full([], $(1), dtype=tl.int64), r10_value)
    }
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(vscatter_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(32768))
    r11_value = (tl.load($(vscatter_ptr) + ((((r11_z / $(128)) % $(256)) * $(128)) + (r11_z % $(128))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r11_value = tl.broadcast_to(r11_value, [$(32768)])
    tl.store($(vto_copy_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(32768))
    r12_value = tl.where(tl.load($(vto_copy_1_ptr) + ((((r12_z / $(128)) % $(256)) * $(128)) + (r12_z % $(128))), dtype=tl.int1), 1.0, 0.0)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(32768)])
    tl.store($(vto_copy_2_ptr) + r12_z, r12_value)
    z = tl.arange($(32768))
    value = tl.load($(vto_copy_2_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_ptr : RegionName := "vadd"
  let vdiv_ptr : RegionName := "vdiv"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vscatter_ptr : RegionName := "vscatter"
  let candidate_vsigmoid_ptr : RegionName := "candidate_vsigmoid"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vsort_value_ptr : RegionName := "candidate_vsort_value"
  let candidate_vsort_index_ptr : RegionName := "candidate_vsort_index"
  let candidate_vscatter_ptr : RegionName := "candidate_vscatter"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  triton {
    r0_z = tl.arange($(32768))
    r0_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r0_z / $(128)) % $(256)) * $(128)) + (r0_z % $(128))), dtype=tl.float32))
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(32768)])
    tl.store($(candidate_vsigmoid_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_lane = tl.arange($(128))
    r1_value = tl.sum(tl.load($(candidate_vsigmoid_ptr) + ((r1_z[:, None] * $(128)) + r1_lane[None, :]), dtype=tl.float32), axis=1)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(candidate_vsum_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_value = (tl.load($(candidate_vsum_1_ptr) + r2_z, dtype=tl.float32) + 0.00000000000000000001)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(candidate_vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(32768))
    r3_value = (tl.load($(candidate_vsigmoid_ptr) + ((((r3_z / $(128)) % $(256)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) / tl.load($(candidate_vadd_ptr) + ((r3_z / $(128)) % $(256)), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(32768)])
    tl.store($(candidate_vdiv_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_lane = tl.arange($(128))
    r4_row_values = tl.load($(candidate_vdiv_ptr) + ((((r4_z[:, None] / $(128)) % $(256)) * $(128)) + (r4_lane[None, :] % $(128))), dtype=tl.float32)
    r4_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r4_other in range($(128)) {
      r4_other_value = tl.load($(candidate_vdiv_ptr) + ((((r4_z[:, None] / $(128)) % $(256)) * $(128)) + (r4_other % $(128))), dtype=tl.float32)
      r4_other_nan = r4_other_value != r4_other_value
      r4_row_nan = r4_row_values != r4_row_values
      r4_strict = (r4_other_nan & (~r4_row_nan)) | (((~r4_other_nan) & (~r4_row_nan)) & (r4_other_value > r4_row_values))
      r4_equal = (r4_other_nan & r4_row_nan) | (r4_other_value == r4_row_values)
      r4_rank = r4_rank + tl.where(r4_strict | (r4_equal & (r4_other < r4_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r4_selected = r4_rank == tl.cast((r4_z % $(128))[:, None], tl.int64)
    r4_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r4_cutoff in range($(128)) {
      r4_value = r4_value + tl.cast(tl.sum(r4_selected & (r4_lane[None, :] > r4_cutoff), axis=1), tl.int64)
    }
    r4_value = tl.load($(candidate_vdiv_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (tl.cast(r4_value, tl.uint32) % $(128))), dtype=tl.float32)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(candidate_vsort_value_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_lane = tl.arange($(128))
    r5_row_values = tl.load($(candidate_vdiv_ptr) + ((((r5_z[:, None] / $(128)) % $(256)) * $(128)) + (r5_lane[None, :] % $(128))), dtype=tl.float32)
    r5_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r5_other in range($(128)) {
      r5_other_value = tl.load($(candidate_vdiv_ptr) + ((((r5_z[:, None] / $(128)) % $(256)) * $(128)) + (r5_other % $(128))), dtype=tl.float32)
      r5_other_nan = r5_other_value != r5_other_value
      r5_row_nan = r5_row_values != r5_row_values
      r5_strict = (r5_other_nan & (~r5_row_nan)) | (((~r5_other_nan) & (~r5_row_nan)) & (r5_other_value > r5_row_values))
      r5_equal = (r5_other_nan & r5_row_nan) | (r5_other_value == r5_row_values)
      r5_rank = r5_rank + tl.where(r5_strict | (r5_equal & (r5_other < r5_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r5_selected = r5_rank == tl.cast((r5_z % $(128))[:, None], tl.int64)
    r5_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r5_cutoff in range($(128)) {
      r5_value = r5_value + tl.cast(tl.sum(r5_selected & (r5_lane[None, :] > r5_cutoff), axis=1), tl.int64)
    }
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(candidate_vsort_index_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32768))
    r6_value = tl.broadcast_to(false, [$(32768)])
    for r6_lane in range($(16)) {
      r6_value = tl.where(tl.load($(candidate_vsort_index_ptr) + ((((r6_z / $(128)) % $(256)) * $(128)) + (r6_lane % $(128))), dtype=tl.int64) == tl.cast((r6_z % $(128)), tl.int64), true, r6_value)
    }
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(candidate_vscatter_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_value = tl.where(tl.load($(candidate_vscatter_ptr) + ((((r7_z / $(128)) % $(256)) * $(128)) + (r7_z % $(128))), dtype=tl.int1), 1.0, 0.0)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32768)])
    tl.store($(candidate_vto_copy_ptr) + r7_z, r7_value)
    z = tl.arange($(32768))
    value = tl.load($(candidate_vto_copy_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

/-- Finite Real specialization. The non-real NaN literal is assigned the unused
Real payload zero. Its connection to `original` is a separate public goal. -/
def realOriginal : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan_ptr : RegionName := "literal_nan"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_ptr : RegionName := "vadd"
  let vdiv_ptr : RegionName := "vdiv"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vscatter_ptr : RegionName := "vscatter"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  triton {
    r0_z = tl.arange($(1))
    r0_value = 0.0
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(32768))
    r1_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r1_z / $(128)) % $(256)) * $(128)) + (r1_z % $(128))), dtype=tl.float32))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(32768)])
    tl.store($(vsigmoid_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_lane = tl.arange($(128))
    r2_value = tl.sum(tl.load($(vsigmoid_ptr) + ((r2_z[:, None] * $(128)) + r2_lane[None, :]), dtype=tl.float32), axis=1)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(vsum_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(256))
    r3_value = (tl.load($(vsum_1_ptr) + r3_z, dtype=tl.float32) + 0.00000000000000000001)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(256)])
    tl.store($(vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = (tl.load($(vsigmoid_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r4_z / $(128)) % $(256)), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(vdiv_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_value = (tl.load($(vdiv_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (r5_z % $(128))), dtype=tl.float32) != tl.load($(vdiv_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (r5_z % $(128))), dtype=tl.float32))
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(visnan_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32768))
    r6_value = tl.where(tl.load($(visnan_ptr) + ((((r6_z / $(128)) % $(256)) * $(128)) + (r6_z % $(128))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(vdiv_ptr) + ((((r6_z / $(128)) % $(256)) * $(128)) + (r6_z % $(128))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(vwhere_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_lane = tl.arange($(128))
    r7_row_values = tl.load($(vwhere_ptr) + ((((r7_z[:, None] / $(128)) % $(256)) * $(128)) + (r7_lane[None, :] % $(128))), dtype=tl.float32)
    r7_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r7_other in range($(128)) {
      r7_other_value = tl.load($(vwhere_ptr) + ((((r7_z[:, None] / $(128)) % $(256)) * $(128)) + (r7_other % $(128))), dtype=tl.float32)
      r7_other_nan = r7_other_value != r7_other_value
      r7_row_nan = r7_row_values != r7_row_values
      r7_strict = (r7_other_nan & (~r7_row_nan)) | (((~r7_other_nan) & (~r7_row_nan)) & (r7_other_value > r7_row_values))
      r7_equal = (r7_other_nan & r7_row_nan) | (r7_other_value == r7_row_values)
      r7_rank = r7_rank + tl.where(r7_strict | (r7_equal & (r7_other < r7_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r7_selected = r7_rank == tl.cast((r7_z % $(128))[:, None], tl.int64)
    r7_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r7_cutoff in range($(128)) {
      r7_value = r7_value + tl.cast(tl.sum(r7_selected & (r7_lane[None, :] > r7_cutoff), axis=1), tl.int64)
    }
    r7_value = tl.load($(vwhere_ptr) + ((((r7_z / $(128)) % $(256)) * $(128)) + (tl.cast(r7_value, tl.uint32) % $(128))), dtype=tl.float32)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32768)])
    tl.store($(vsort_value_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32768))
    r8_lane = tl.arange($(128))
    r8_row_values = tl.load($(vwhere_ptr) + ((((r8_z[:, None] / $(128)) % $(256)) * $(128)) + (r8_lane[None, :] % $(128))), dtype=tl.float32)
    r8_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r8_other in range($(128)) {
      r8_other_value = tl.load($(vwhere_ptr) + ((((r8_z[:, None] / $(128)) % $(256)) * $(128)) + (r8_other % $(128))), dtype=tl.float32)
      r8_other_nan = r8_other_value != r8_other_value
      r8_row_nan = r8_row_values != r8_row_values
      r8_strict = (r8_other_nan & (~r8_row_nan)) | (((~r8_other_nan) & (~r8_row_nan)) & (r8_other_value > r8_row_values))
      r8_equal = (r8_other_nan & r8_row_nan) | (r8_other_value == r8_row_values)
      r8_rank = r8_rank + tl.where(r8_strict | (r8_equal & (r8_other < r8_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r8_selected = r8_rank == tl.cast((r8_z % $(128))[:, None], tl.int64)
    r8_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r8_cutoff in range($(128)) {
      r8_value = r8_value + tl.cast(tl.sum(r8_selected & (r8_lane[None, :] > r8_cutoff), axis=1), tl.int64)
    }
    r8_value = tl.broadcast_to(r8_value, [$(32768)])
    tl.store($(vsort_index_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(32768))
    r9_value = tl.full([], $(0), dtype=tl.int64)
    r9_value = tl.broadcast_to(r9_value, [$(32768)])
    tl.store($(vto_copy_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(32768))
    r10_value = tl.broadcast_to(tl.load($(vto_copy_ptr) + ((((r10_z / $(128)) % $(256)) * $(128)) + (r10_z % $(128))), dtype=tl.int64), [$(32768)])
    for r10_lane in range($(16)) {
      r10_value = tl.where(tl.load($(vsort_index_ptr) + ((((r10_z / $(128)) % $(256)) * $(128)) + (r10_lane % $(128))), dtype=tl.int64) == tl.cast((r10_z % $(128)), tl.int64), tl.full([], $(1), dtype=tl.int64), r10_value)
    }
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(vscatter_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(32768))
    r11_value = (tl.load($(vscatter_ptr) + ((((r11_z / $(128)) % $(256)) * $(128)) + (r11_z % $(128))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r11_value = tl.broadcast_to(r11_value, [$(32768)])
    tl.store($(vto_copy_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(32768))
    r12_value = tl.where(tl.load($(vto_copy_1_ptr) + ((((r12_z / $(128)) % $(256)) * $(128)) + (r12_z % $(128))), dtype=tl.int1), 1.0, 0.0)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(32768)])
    tl.store($(vto_copy_2_ptr) + r12_z, r12_value)
    z = tl.arange($(32768))
    value = tl.load($(vto_copy_2_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss_routing_map
