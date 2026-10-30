import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_triton_topk
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
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vgather_ptr : RegionName := "vgather"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    r0_value = r0_value.to(tl.bfloat16)
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1048576))
    r1_value = (tl.load($(input0_ptr) + ((((r1_z / $(4096)) % $(256)) * $(4096)) + (r1_z % $(4096))), dtype=tl.bfloat16) != tl.load($(input0_ptr) + ((((r1_z / $(4096)) % $(256)) * $(4096)) + (r1_z % $(4096))), dtype=tl.bfloat16))
    r1_value = tl.broadcast_to(r1_value, [$(1048576)])
    tl.store($(visnan_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(1048576))
    r2_value = tl.where(tl.load($(visnan_ptr) + ((((r2_z / $(4096)) % $(256)) * $(4096)) + (r2_z % $(4096))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.bfloat16), tl.load($(input0_ptr) + ((((r2_z / $(4096)) % $(256)) * $(4096)) + (r2_z % $(4096))), dtype=tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(1048576)])
    tl.store($(vwhere_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1048576))
    r3_lane = tl.arange($(4096))
    r3_row_values = tl.load($(vwhere_ptr) + ((((r3_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r3_lane[None, :] % $(4096))), dtype=tl.bfloat16)
    r3_rank = tl.full([$(1048576), $(4096)], $(0), dtype=tl.int64)
    for r3_other in range($(4096)) {
      r3_other_value = tl.load($(vwhere_ptr) + ((((r3_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r3_other % $(4096))), dtype=tl.bfloat16)
      r3_other_nan = r3_other_value != r3_other_value
      r3_row_nan = r3_row_values != r3_row_values
      r3_strict = (r3_other_nan & (~r3_row_nan)) | (((~r3_other_nan) & (~r3_row_nan)) & (r3_other_value > r3_row_values))
      r3_equal = (r3_other_nan & r3_row_nan) | (r3_other_value == r3_row_values)
      r3_rank = r3_rank + tl.where(r3_strict | (r3_equal & (r3_other < r3_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r3_selected = r3_rank == tl.cast((r3_z % $(4096))[:, None], tl.int64)
    r3_value = tl.full([$(1048576)], $(0), dtype=tl.int64)
    for r3_cutoff in range($(4096)) {
      r3_value = r3_value + tl.cast(tl.sum(r3_selected & (r3_lane[None, :] > r3_cutoff), axis=1), tl.int64)
    }
    r3_value = tl.load($(vwhere_ptr) + ((((r3_z / $(4096)) % $(256)) * $(4096)) + (tl.cast(r3_value, tl.uint32) % $(4096))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(1048576)])
    tl.store($(vsort_value_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(1048576))
    r4_lane = tl.arange($(4096))
    r4_row_values = tl.load($(vwhere_ptr) + ((((r4_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r4_lane[None, :] % $(4096))), dtype=tl.bfloat16)
    r4_rank = tl.full([$(1048576), $(4096)], $(0), dtype=tl.int64)
    for r4_other in range($(4096)) {
      r4_other_value = tl.load($(vwhere_ptr) + ((((r4_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r4_other % $(4096))), dtype=tl.bfloat16)
      r4_other_nan = r4_other_value != r4_other_value
      r4_row_nan = r4_row_values != r4_row_values
      r4_strict = (r4_other_nan & (~r4_row_nan)) | (((~r4_other_nan) & (~r4_row_nan)) & (r4_other_value > r4_row_values))
      r4_equal = (r4_other_nan & r4_row_nan) | (r4_other_value == r4_row_values)
      r4_rank = r4_rank + tl.where(r4_strict | (r4_equal & (r4_other < r4_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r4_selected = r4_rank == tl.cast((r4_z % $(4096))[:, None], tl.int64)
    r4_value = tl.full([$(1048576)], $(0), dtype=tl.int64)
    for r4_cutoff in range($(4096)) {
      r4_value = r4_value + tl.cast(tl.sum(r4_selected & (r4_lane[None, :] > r4_cutoff), axis=1), tl.int64)
    }
    r4_value = tl.broadcast_to(r4_value, [$(1048576)])
    tl.store($(vsort_index_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2048))
    r5_value = tl.load($(input0_ptr) + ((((r5_z / $(8)) % $(256)) * $(4096)) + (tl.load($(vsort_index_ptr) + ((((r5_z / $(8)) % $(256)) * $(4096)) + (r5_z % $(8))), dtype=tl.int64) % $(4096))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(2048)])
    tl.store($(vgather_ptr) + r5_z, r5_value)
    z = tl.arange($(2048))
    value = tl.load($(vgather_ptr) + ((((z / $(8)) % $(256)) * $(8)) + (z % $(8))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2048)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan_ptr : RegionName := "literal_nan"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vgather_ptr : RegionName := "vgather"
  let candidate_literal_nan_ptr : RegionName := "candidate_literal_nan"
  let candidate_visnan_ptr : RegionName := "candidate_visnan"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vsort_value_ptr : RegionName := "candidate_vsort_value"
  let candidate_vsort_index_ptr : RegionName := "candidate_vsort_index"
  let candidate_vgather_ptr : RegionName := "candidate_vgather"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    r0_value = r0_value.to(tl.bfloat16)
    tl.store($(candidate_literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1048576))
    r1_value = (tl.load($(input0_ptr) + ((((r1_z / $(4096)) % $(256)) * $(4096)) + (r1_z % $(4096))), dtype=tl.bfloat16) != tl.load($(input0_ptr) + ((((r1_z / $(4096)) % $(256)) * $(4096)) + (r1_z % $(4096))), dtype=tl.bfloat16))
    r1_value = tl.broadcast_to(r1_value, [$(1048576)])
    tl.store($(candidate_visnan_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(1048576))
    r2_value = tl.where(tl.load($(candidate_visnan_ptr) + ((((r2_z / $(4096)) % $(256)) * $(4096)) + (r2_z % $(4096))), dtype=tl.int1), tl.load($(candidate_literal_nan_ptr) + $(0), dtype=tl.bfloat16), tl.load($(input0_ptr) + ((((r2_z / $(4096)) % $(256)) * $(4096)) + (r2_z % $(4096))), dtype=tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(1048576)])
    tl.store($(candidate_vwhere_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1048576))
    r3_lane = tl.arange($(4096))
    r3_row_values = tl.load($(candidate_vwhere_ptr) + ((((r3_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r3_lane[None, :] % $(4096))), dtype=tl.bfloat16)
    r3_rank = tl.full([$(1048576), $(4096)], $(0), dtype=tl.int64)
    for r3_other in range($(4096)) {
      r3_other_value = tl.load($(candidate_vwhere_ptr) + ((((r3_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r3_other % $(4096))), dtype=tl.bfloat16)
      r3_other_nan = r3_other_value != r3_other_value
      r3_row_nan = r3_row_values != r3_row_values
      r3_strict = (r3_other_nan & (~r3_row_nan)) | (((~r3_other_nan) & (~r3_row_nan)) & (r3_other_value > r3_row_values))
      r3_equal = (r3_other_nan & r3_row_nan) | (r3_other_value == r3_row_values)
      r3_rank = r3_rank + tl.where(r3_strict | (r3_equal & (r3_other < r3_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r3_selected = r3_rank == tl.cast((r3_z % $(4096))[:, None], tl.int64)
    r3_value = tl.full([$(1048576)], $(0), dtype=tl.int64)
    for r3_cutoff in range($(4096)) {
      r3_value = r3_value + tl.cast(tl.sum(r3_selected & (r3_lane[None, :] > r3_cutoff), axis=1), tl.int64)
    }
    r3_value = tl.load($(candidate_vwhere_ptr) + ((((r3_z / $(4096)) % $(256)) * $(4096)) + (tl.cast(r3_value, tl.uint32) % $(4096))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(1048576)])
    tl.store($(candidate_vsort_value_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(1048576))
    r4_lane = tl.arange($(4096))
    r4_row_values = tl.load($(candidate_vwhere_ptr) + ((((r4_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r4_lane[None, :] % $(4096))), dtype=tl.bfloat16)
    r4_rank = tl.full([$(1048576), $(4096)], $(0), dtype=tl.int64)
    for r4_other in range($(4096)) {
      r4_other_value = tl.load($(candidate_vwhere_ptr) + ((((r4_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r4_other % $(4096))), dtype=tl.bfloat16)
      r4_other_nan = r4_other_value != r4_other_value
      r4_row_nan = r4_row_values != r4_row_values
      r4_strict = (r4_other_nan & (~r4_row_nan)) | (((~r4_other_nan) & (~r4_row_nan)) & (r4_other_value > r4_row_values))
      r4_equal = (r4_other_nan & r4_row_nan) | (r4_other_value == r4_row_values)
      r4_rank = r4_rank + tl.where(r4_strict | (r4_equal & (r4_other < r4_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r4_selected = r4_rank == tl.cast((r4_z % $(4096))[:, None], tl.int64)
    r4_value = tl.full([$(1048576)], $(0), dtype=tl.int64)
    for r4_cutoff in range($(4096)) {
      r4_value = r4_value + tl.cast(tl.sum(r4_selected & (r4_lane[None, :] > r4_cutoff), axis=1), tl.int64)
    }
    r4_value = tl.broadcast_to(r4_value, [$(1048576)])
    tl.store($(candidate_vsort_index_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2048))
    r5_value = tl.load($(input0_ptr) + ((((r5_z / $(8)) % $(256)) * $(4096)) + (tl.load($(candidate_vsort_index_ptr) + ((((r5_z / $(8)) % $(256)) * $(4096)) + (r5_z % $(8))), dtype=tl.int64) % $(4096))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(2048)])
    tl.store($(candidate_vgather_ptr) + r5_z, r5_value)
    z = tl.arange($(2048))
    value = tl.load($(candidate_vgather_ptr) + ((((z / $(8)) % $(256)) * $(8)) + (z % $(8))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2048)])
    tl.store($(out_ptr) + z, value)
  }

/-- Finite Real specialization. The non-real NaN literal is assigned the unused
Real payload zero. Its connection to `original` is a separate public goal. -/
def realOriginal : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan_ptr : RegionName := "literal_nan"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vgather_ptr : RegionName := "vgather"
  triton {
    r0_z = tl.arange($(1))
    r0_value = 0.0
    r0_value = r0_value.to(tl.bfloat16)
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1048576))
    r1_value = (tl.load($(input0_ptr) + ((((r1_z / $(4096)) % $(256)) * $(4096)) + (r1_z % $(4096))), dtype=tl.bfloat16) != tl.load($(input0_ptr) + ((((r1_z / $(4096)) % $(256)) * $(4096)) + (r1_z % $(4096))), dtype=tl.bfloat16))
    r1_value = tl.broadcast_to(r1_value, [$(1048576)])
    tl.store($(visnan_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(1048576))
    r2_value = tl.where(tl.load($(visnan_ptr) + ((((r2_z / $(4096)) % $(256)) * $(4096)) + (r2_z % $(4096))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.bfloat16), tl.load($(input0_ptr) + ((((r2_z / $(4096)) % $(256)) * $(4096)) + (r2_z % $(4096))), dtype=tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(1048576)])
    tl.store($(vwhere_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1048576))
    r3_lane = tl.arange($(4096))
    r3_row_values = tl.load($(vwhere_ptr) + ((((r3_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r3_lane[None, :] % $(4096))), dtype=tl.bfloat16)
    r3_rank = tl.full([$(1048576), $(4096)], $(0), dtype=tl.int64)
    for r3_other in range($(4096)) {
      r3_other_value = tl.load($(vwhere_ptr) + ((((r3_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r3_other % $(4096))), dtype=tl.bfloat16)
      r3_other_nan = r3_other_value != r3_other_value
      r3_row_nan = r3_row_values != r3_row_values
      r3_strict = (r3_other_nan & (~r3_row_nan)) | (((~r3_other_nan) & (~r3_row_nan)) & (r3_other_value > r3_row_values))
      r3_equal = (r3_other_nan & r3_row_nan) | (r3_other_value == r3_row_values)
      r3_rank = r3_rank + tl.where(r3_strict | (r3_equal & (r3_other < r3_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r3_selected = r3_rank == tl.cast((r3_z % $(4096))[:, None], tl.int64)
    r3_value = tl.full([$(1048576)], $(0), dtype=tl.int64)
    for r3_cutoff in range($(4096)) {
      r3_value = r3_value + tl.cast(tl.sum(r3_selected & (r3_lane[None, :] > r3_cutoff), axis=1), tl.int64)
    }
    r3_value = tl.load($(vwhere_ptr) + ((((r3_z / $(4096)) % $(256)) * $(4096)) + (tl.cast(r3_value, tl.uint32) % $(4096))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(1048576)])
    tl.store($(vsort_value_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(1048576))
    r4_lane = tl.arange($(4096))
    r4_row_values = tl.load($(vwhere_ptr) + ((((r4_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r4_lane[None, :] % $(4096))), dtype=tl.bfloat16)
    r4_rank = tl.full([$(1048576), $(4096)], $(0), dtype=tl.int64)
    for r4_other in range($(4096)) {
      r4_other_value = tl.load($(vwhere_ptr) + ((((r4_z[:, None] / $(4096)) % $(256)) * $(4096)) + (r4_other % $(4096))), dtype=tl.bfloat16)
      r4_other_nan = r4_other_value != r4_other_value
      r4_row_nan = r4_row_values != r4_row_values
      r4_strict = (r4_other_nan & (~r4_row_nan)) | (((~r4_other_nan) & (~r4_row_nan)) & (r4_other_value > r4_row_values))
      r4_equal = (r4_other_nan & r4_row_nan) | (r4_other_value == r4_row_values)
      r4_rank = r4_rank + tl.where(r4_strict | (r4_equal & (r4_other < r4_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r4_selected = r4_rank == tl.cast((r4_z % $(4096))[:, None], tl.int64)
    r4_value = tl.full([$(1048576)], $(0), dtype=tl.int64)
    for r4_cutoff in range($(4096)) {
      r4_value = r4_value + tl.cast(tl.sum(r4_selected & (r4_lane[None, :] > r4_cutoff), axis=1), tl.int64)
    }
    r4_value = tl.broadcast_to(r4_value, [$(1048576)])
    tl.store($(vsort_index_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2048))
    r5_value = tl.load($(input0_ptr) + ((((r5_z / $(8)) % $(256)) * $(4096)) + (tl.load($(vsort_index_ptr) + ((((r5_z / $(8)) % $(256)) * $(4096)) + (r5_z % $(8))), dtype=tl.int64) % $(4096))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(2048)])
    tl.store($(vgather_ptr) + r5_z, r5_value)
    z = tl.arange($(2048))
    value = tl.load($(vgather_ptr) + ((((z / $(8)) % $(256)) * $(8)) + (z % $(8))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2048)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_triton_topk
