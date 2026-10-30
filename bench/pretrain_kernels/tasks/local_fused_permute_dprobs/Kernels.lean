import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_permute_dprobs
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vtopk_value_ptr : RegionName := "vtopk_value"
  let vtopk_index_ptr : RegionName := "vtopk_index"
  let vscatter_ptr : RegionName := "vscatter"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vindex_put_ptr : RegionName := "vindex_put"
  triton {
    r0_z = tl.arange($(1536))
    r0_lane = tl.arange($(128))
    r0_row_values = tl.load($(parameter0_ptr) + ((((r0_z[:, None] / $(6)) % $(256)) * $(128)) + (r0_lane[None, :] % $(128))), dtype=tl.float32)
    r0_rank = tl.full([$(1536), $(128)], $(0), dtype=tl.int64)
    for r0_other in range($(128)) {
      r0_other_value = tl.load($(parameter0_ptr) + ((((r0_z[:, None] / $(6)) % $(256)) * $(128)) + (r0_other % $(128))), dtype=tl.float32)
      r0_other_nan = r0_other_value != r0_other_value
      r0_row_nan = r0_row_values != r0_row_values
      r0_strict = (r0_other_nan & (~r0_row_nan)) | (((~r0_other_nan) & (~r0_row_nan)) & (r0_other_value > r0_row_values))
      r0_equal = (r0_other_nan & r0_row_nan) | (r0_other_value == r0_row_values)
      r0_rank = r0_rank + tl.where(r0_strict | (r0_equal & (r0_other < r0_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r0_selected = r0_rank == tl.cast((r0_z % $(6))[:, None], tl.int64)
    r0_value = tl.full([$(1536)], $(0), dtype=tl.int64)
    for r0_cutoff in range($(128)) {
      r0_value = r0_value + tl.cast(tl.sum(r0_selected & (r0_lane[None, :] > r0_cutoff), axis=1), tl.int64)
    }
    r0_value = tl.load($(parameter0_ptr) + ((((r0_z / $(6)) % $(256)) * $(128)) + (tl.cast(r0_value, tl.uint32) % $(128))), dtype=tl.float32)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(1536)])
    tl.store($(vtopk_value_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1536))
    r1_lane = tl.arange($(128))
    r1_row_values = tl.load($(parameter0_ptr) + ((((r1_z[:, None] / $(6)) % $(256)) * $(128)) + (r1_lane[None, :] % $(128))), dtype=tl.float32)
    r1_rank = tl.full([$(1536), $(128)], $(0), dtype=tl.int64)
    for r1_other in range($(128)) {
      r1_other_value = tl.load($(parameter0_ptr) + ((((r1_z[:, None] / $(6)) % $(256)) * $(128)) + (r1_other % $(128))), dtype=tl.float32)
      r1_other_nan = r1_other_value != r1_other_value
      r1_row_nan = r1_row_values != r1_row_values
      r1_strict = (r1_other_nan & (~r1_row_nan)) | (((~r1_other_nan) & (~r1_row_nan)) & (r1_other_value > r1_row_values))
      r1_equal = (r1_other_nan & r1_row_nan) | (r1_other_value == r1_row_values)
      r1_rank = r1_rank + tl.where(r1_strict | (r1_equal & (r1_other < r1_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r1_selected = r1_rank == tl.cast((r1_z % $(6))[:, None], tl.int64)
    r1_value = tl.full([$(1536)], $(0), dtype=tl.int64)
    for r1_cutoff in range($(128)) {
      r1_value = r1_value + tl.cast(tl.sum(r1_selected & (r1_lane[None, :] > r1_cutoff), axis=1), tl.int64)
    }
    r1_value = tl.broadcast_to(r1_value, [$(1536)])
    tl.store($(vtopk_index_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32768))
    r2_value = tl.broadcast_to(false, [$(32768)])
    for r2_lane in range($(6)) {
      r2_value = tl.where(tl.load($(vtopk_index_ptr) + ((((r2_z / $(128)) % $(256)) * $(6)) + (r2_lane % $(6))), dtype=tl.int64) == tl.cast((r2_z % $(128)), tl.int64), true, r2_value)
    }
    r2_value = tl.broadcast_to(r2_value, [$(32768)])
    tl.store($(vscatter_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1536))
    r3_value = tl.load($(input0_ptr) + r3_z, dtype=tl.float64)
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(1536)])
    tl.store($(vto_copy_3_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_lane = tl.arange($(32768))
    r4_row_values = tl.load($(vscatter_ptr) + (((r4_lane[None, :] % $(256)) * $(128)) + ((r4_lane[None, :] / $(256)) % $(128))), dtype=tl.int1)
    r4_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r4_other in range($(32768)) {
      r4_rank = r4_rank + tl.where(((tl.load($(vscatter_ptr) + (((r4_other % $(256)) * $(128)) + ((r4_other / $(256)) % $(128))), dtype=tl.int1) & (~r4_row_values)) | (((tl.load($(vscatter_ptr) + (((r4_other % $(256)) * $(128)) + ((r4_other / $(256)) % $(128))), dtype=tl.int1) & r4_row_values) | ((~tl.load($(vscatter_ptr) + (((r4_other % $(256)) * $(128)) + ((r4_other / $(256)) % $(128))), dtype=tl.int1)) & (~r4_row_values))) & (r4_other < r4_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r4_selected = r4_rank == tl.cast(r4_z[:, None], tl.int64)
    r4_value = tl.sum(r4_selected & r4_row_values, axis=1) > $(0)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(vsort_value_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_lane = tl.arange($(32768))
    r5_row_values = tl.load($(vscatter_ptr) + (((r5_lane[None, :] % $(256)) * $(128)) + ((r5_lane[None, :] / $(256)) % $(128))), dtype=tl.int1)
    r5_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r5_other in range($(32768)) {
      r5_rank = r5_rank + tl.where(((tl.load($(vscatter_ptr) + (((r5_other % $(256)) * $(128)) + ((r5_other / $(256)) % $(128))), dtype=tl.int1) & (~r5_row_values)) | (((tl.load($(vscatter_ptr) + (((r5_other % $(256)) * $(128)) + ((r5_other / $(256)) % $(128))), dtype=tl.int1) & r5_row_values) | ((~tl.load($(vscatter_ptr) + (((r5_other % $(256)) * $(128)) + ((r5_other / $(256)) % $(128))), dtype=tl.int1)) & (~r5_row_values))) & (r5_other < r5_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r5_selected = r5_rank == tl.cast(r5_z[:, None], tl.int64)
    r5_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r5_cutoff in range($(32768)) {
      r5_value = r5_value + tl.cast(tl.sum(r5_selected & (r5_lane[None, :] > r5_cutoff), axis=1), tl.int64)
    }
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(vsort_index_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32768))
    r6_value = tl.cast(0.0, tl.bfloat16)
    for r6_lane in range($(1536)) {
      r6_value = tl.where(tl.load($(vsort_index_ptr) + (r6_lane % $(32768)), dtype=tl.int64) == r6_z, r6_value + tl.load($(vto_copy_3_ptr) + (r6_lane % $(1536)), dtype=tl.bfloat16), r6_value)
    }
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(vindex_put_ptr) + r6_z, r6_value)
    z = tl.arange($(32768))
    value = tl.load($(vindex_put_ptr) + (((z % $(128)) * $(256)) + ((z / $(128)) % $(256))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vtopk_value_ptr : RegionName := "vtopk_value"
  let vtopk_index_ptr : RegionName := "vtopk_index"
  let vscatter_ptr : RegionName := "vscatter"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let candidate_vtopk_value_ptr : RegionName := "candidate_vtopk_value"
  let candidate_vtopk_index_ptr : RegionName := "candidate_vtopk_index"
  let candidate_vscatter_ptr : RegionName := "candidate_vscatter"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vcumsum_ptr : RegionName := "candidate_vcumsum"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vto_copy_7_ptr : RegionName := "candidate_vto_copy_7"
  let candidate_vcumsum_1_ptr : RegionName := "candidate_vcumsum_1"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vwhere_1_ptr : RegionName := "candidate_vwhere_1"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vgather_ptr : RegionName := "candidate_vgather"
  let candidate_vwhere_4_ptr : RegionName := "candidate_vwhere_4"
  triton {
    r0_z = tl.arange($(1536))
    r0_lane = tl.arange($(128))
    r0_row_values = tl.load($(parameter0_ptr) + ((((r0_z[:, None] / $(6)) % $(256)) * $(128)) + (r0_lane[None, :] % $(128))), dtype=tl.float32)
    r0_rank = tl.full([$(1536), $(128)], $(0), dtype=tl.int64)
    for r0_other in range($(128)) {
      r0_other_value = tl.load($(parameter0_ptr) + ((((r0_z[:, None] / $(6)) % $(256)) * $(128)) + (r0_other % $(128))), dtype=tl.float32)
      r0_other_nan = r0_other_value != r0_other_value
      r0_row_nan = r0_row_values != r0_row_values
      r0_strict = (r0_other_nan & (~r0_row_nan)) | (((~r0_other_nan) & (~r0_row_nan)) & (r0_other_value > r0_row_values))
      r0_equal = (r0_other_nan & r0_row_nan) | (r0_other_value == r0_row_values)
      r0_rank = r0_rank + tl.where(r0_strict | (r0_equal & (r0_other < r0_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r0_selected = r0_rank == tl.cast((r0_z % $(6))[:, None], tl.int64)
    r0_value = tl.full([$(1536)], $(0), dtype=tl.int64)
    for r0_cutoff in range($(128)) {
      r0_value = r0_value + tl.cast(tl.sum(r0_selected & (r0_lane[None, :] > r0_cutoff), axis=1), tl.int64)
    }
    r0_value = tl.load($(parameter0_ptr) + ((((r0_z / $(6)) % $(256)) * $(128)) + (tl.cast(r0_value, tl.uint32) % $(128))), dtype=tl.float32)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(1536)])
    tl.store($(candidate_vtopk_value_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1536))
    r1_lane = tl.arange($(128))
    r1_row_values = tl.load($(parameter0_ptr) + ((((r1_z[:, None] / $(6)) % $(256)) * $(128)) + (r1_lane[None, :] % $(128))), dtype=tl.float32)
    r1_rank = tl.full([$(1536), $(128)], $(0), dtype=tl.int64)
    for r1_other in range($(128)) {
      r1_other_value = tl.load($(parameter0_ptr) + ((((r1_z[:, None] / $(6)) % $(256)) * $(128)) + (r1_other % $(128))), dtype=tl.float32)
      r1_other_nan = r1_other_value != r1_other_value
      r1_row_nan = r1_row_values != r1_row_values
      r1_strict = (r1_other_nan & (~r1_row_nan)) | (((~r1_other_nan) & (~r1_row_nan)) & (r1_other_value > r1_row_values))
      r1_equal = (r1_other_nan & r1_row_nan) | (r1_other_value == r1_row_values)
      r1_rank = r1_rank + tl.where(r1_strict | (r1_equal & (r1_other < r1_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r1_selected = r1_rank == tl.cast((r1_z % $(6))[:, None], tl.int64)
    r1_value = tl.full([$(1536)], $(0), dtype=tl.int64)
    for r1_cutoff in range($(128)) {
      r1_value = r1_value + tl.cast(tl.sum(r1_selected & (r1_lane[None, :] > r1_cutoff), axis=1), tl.int64)
    }
    r1_value = tl.broadcast_to(r1_value, [$(1536)])
    tl.store($(candidate_vtopk_index_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32768))
    r2_value = tl.broadcast_to(false, [$(32768)])
    for r2_lane in range($(6)) {
      r2_value = tl.where(tl.load($(candidate_vtopk_index_ptr) + ((((r2_z / $(128)) % $(256)) * $(6)) + (r2_lane % $(6))), dtype=tl.int64) == tl.cast((r2_z % $(128)), tl.int64), true, r2_value)
    }
    r2_value = tl.broadcast_to(r2_value, [$(32768)])
    tl.store($(candidate_vscatter_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1536))
    r3_value = tl.load($(input0_ptr) + r3_z, dtype=tl.float64)
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(1536)])
    tl.store($(candidate_vto_copy_3_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = tl.where(tl.load($(candidate_vscatter_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(candidate_vto_copy_4_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_value = (tl.load($(candidate_vto_copy_4_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (r5_z % $(128))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(candidate_vto_copy_5_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32768))
    r6_value = tl.where(tl.load($(candidate_vto_copy_5_ptr) + ((((r6_z / $(128)) % $(256)) * $(128)) + (r6_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(candidate_vto_copy_6_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(128))
    r7_value = tl.full([$(128)], $(0), dtype=tl.int64)
    for r7_lane in range($(256)) {
      r7_value = r7_value + tl.load($(candidate_vto_copy_6_ptr) + ((r7_lane * $(128)) + r7_z), dtype=tl.int64)
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
    r10_value = tl.where(tl.load($(candidate_vto_copy_5_ptr) + ((((r10_z / $(128)) % $(256)) * $(128)) + (r10_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(candidate_vto_copy_7_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(32768))
    r11_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(32768)])
    for r11_lane in range($(256)) {
      r11_value = tl.where(r11_lane <= ((r11_z / $(128)) % $(256)), (r11_value + tl.load($(candidate_vto_copy_7_ptr) + (((r11_lane % $(256)) * $(128)) + (r11_z % $(128))), dtype=tl.int64)), r11_value)
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
    r14_value = tl.where(tl.load($(candidate_vto_copy_5_ptr) + ((((r14_z / $(128)) % $(256)) * $(128)) + (r14_z % $(128))), dtype=tl.int1), tl.load($(candidate_vadd_ptr) + ((((r14_z / $(128)) % $(256)) * $(128)) + (r14_z % $(128))), dtype=tl.int64), (-tl.full([], $(1), dtype=tl.int64)))
    r14_value = tl.broadcast_to(r14_value, [$(32768)])
    tl.store($(candidate_vwhere_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(32768))
    r15_value = tl.where(tl.load($(candidate_vto_copy_5_ptr) + ((((r15_z / $(128)) % $(256)) * $(128)) + (r15_z % $(128))), dtype=tl.int1), tl.load($(candidate_vwhere_ptr) + ((((r15_z / $(128)) % $(256)) * $(128)) + (r15_z % $(128))), dtype=tl.int64), tl.full([], $(1536), dtype=tl.int64))
    r15_value = tl.broadcast_to(r15_value, [$(32768)])
    tl.store($(candidate_vwhere_1_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(1537))
    r16_value = tl.where(((r16_z >= $(0)) & (r16_z < $(1536)) & ((r16_z - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_3_ptr) + ((r16_z - $(0)) % $(1536)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r16_value = r16_value.to(tl.bfloat16)
    r16_value = tl.broadcast_to(r16_value, [$(1537)])
    tl.store($(candidate_vslice_backward_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_value = tl.load($(candidate_vslice_backward_ptr) + (tl.load($(candidate_vwhere_1_ptr) + ((((r17_z / $(128)) % $(256)) * $(128)) + (r17_z % $(128))), dtype=tl.int64) % $(1537)), dtype=tl.bfloat16)
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(candidate_vgather_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = tl.where(tl.load($(candidate_vto_copy_5_ptr) + ((((r18_z / $(128)) % $(256)) * $(128)) + (r18_z % $(128))), dtype=tl.int1), tl.load($(candidate_vgather_ptr) + ((((r18_z / $(128)) % $(256)) * $(128)) + (r18_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(candidate_vwhere_4_ptr) + r18_z, r18_value)
    z = tl.arange($(32768))
    value = tl.load($(candidate_vwhere_4_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_permute_dprobs
