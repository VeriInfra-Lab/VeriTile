import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_unpermute_dmerging_probs
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vtopk_value_ptr : RegionName := "vtopk_value"
  let vtopk_index_ptr : RegionName := "vtopk_index"
  let vscatter_ptr : RegionName := "vscatter"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vremainder_ptr : RegionName := "vremainder"
  let vindex_select_ptr : RegionName := "vindex_select"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vto_copy_7_ptr : RegionName := "vto_copy_7"
  let vsort_1_value_ptr : RegionName := "vsort_1_value"
  let vsort_1_index_ptr : RegionName := "vsort_1_index"
  let vto_copy_8_ptr : RegionName := "vto_copy_8"
  let vto_copy_9_ptr : RegionName := "vto_copy_9"
  let vgather_ptr : RegionName := "vgather"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vindex_put_ptr : RegionName := "vindex_put"
  let vto_copy_10_ptr : RegionName := "vto_copy_10"
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
    r3_z = tl.arange($(32768))
    r3_lane = tl.arange($(32768))
    r3_row_values = tl.load($(vscatter_ptr) + (((r3_lane[None, :] % $(256)) * $(128)) + ((r3_lane[None, :] / $(256)) % $(128))), dtype=tl.int1)
    r3_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r3_other in range($(32768)) {
      r3_rank = r3_rank + tl.where(((tl.load($(vscatter_ptr) + (((r3_other % $(256)) * $(128)) + ((r3_other / $(256)) % $(128))), dtype=tl.int1) & (~r3_row_values)) | (((tl.load($(vscatter_ptr) + (((r3_other % $(256)) * $(128)) + ((r3_other / $(256)) % $(128))), dtype=tl.int1) & r3_row_values) | ((~tl.load($(vscatter_ptr) + (((r3_other % $(256)) * $(128)) + ((r3_other / $(256)) % $(128))), dtype=tl.int1)) & (~r3_row_values))) & (r3_other < r3_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r3_selected = r3_rank == tl.cast(r3_z[:, None], tl.int64)
    r3_value = tl.sum(r3_selected & r3_row_values, axis=1) > $(0)
    r3_value = tl.broadcast_to(r3_value, [$(32768)])
    tl.store($(vsort_value_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_lane = tl.arange($(32768))
    r4_row_values = tl.load($(vscatter_ptr) + (((r4_lane[None, :] % $(256)) * $(128)) + ((r4_lane[None, :] / $(256)) % $(128))), dtype=tl.int1)
    r4_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r4_other in range($(32768)) {
      r4_rank = r4_rank + tl.where(((tl.load($(vscatter_ptr) + (((r4_other % $(256)) * $(128)) + ((r4_other / $(256)) % $(128))), dtype=tl.int1) & (~r4_row_values)) | (((tl.load($(vscatter_ptr) + (((r4_other % $(256)) * $(128)) + ((r4_other / $(256)) % $(128))), dtype=tl.int1) & r4_row_values) | ((~tl.load($(vscatter_ptr) + (((r4_other % $(256)) * $(128)) + ((r4_other / $(256)) % $(128))), dtype=tl.int1)) & (~r4_row_values))) & (r4_other < r4_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r4_selected = r4_rank == tl.cast(r4_z[:, None], tl.int64)
    r4_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r4_cutoff in range($(32768)) {
      r4_value = r4_value + tl.cast(tl.sum(r4_selected & (r4_lane[None, :] > r4_cutoff), axis=1), tl.int64)
    }
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(vsort_index_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(1536))
    r5_value = (tl.load($(vsort_index_ptr) + r5_z, dtype=tl.int64) % tl.full([], $(256), dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(1536)])
    tl.store($(vremainder_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(3145728))
    r6_value = tl.load($(parameter1_ptr) + (((tl.load($(vremainder_ptr) + ((r6_z / $(2048)) % $(1536)), dtype=tl.int64) % $(256)) * $(2048)) + (r6_z % $(2048))), dtype=tl.bfloat16)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(3145728)])
    tl.store($(vindex_select_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(3145728))
    r7_value = tl.load($(vindex_select_ptr) + ((((r7_z / $(2048)) % $(1536)) * $(2048)) + (r7_z % $(2048))), dtype=tl.bfloat16)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(3145728)])
    tl.store($(vto_copy_6_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32768))
    r8_value = tl.where(tl.load($(vscatter_ptr) + (((r8_z % $(256)) * $(128)) + ((r8_z / $(256)) % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r8_value = tl.broadcast_to(r8_value, [$(32768)])
    tl.store($(vto_copy_7_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(32768))
    r9_lane = tl.arange($(32768))
    r9_row_values = tl.load($(vto_copy_7_ptr) + (r9_lane[None, :] % $(32768)), dtype=tl.int64)
    r9_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r9_other in range($(32768)) {
      r9_rank = r9_rank + tl.where(((tl.load($(vto_copy_7_ptr) + (r9_other % $(32768)), dtype=tl.int64) > r9_row_values) | ((tl.load($(vto_copy_7_ptr) + (r9_other % $(32768)), dtype=tl.int64) == r9_row_values) & (r9_other < r9_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r9_selected = r9_rank == tl.cast(r9_z[:, None], tl.int64)
    r9_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r9_cutoff in range($(32768)) {
      r9_value = r9_value + tl.cast(tl.sum(r9_selected & (r9_lane[None, :] > r9_cutoff), axis=1), tl.int64)
    }
    r9_value = tl.load($(vto_copy_7_ptr) + tl.cast(r9_value, tl.uint32), dtype=tl.int64)
    r9_value = tl.broadcast_to(r9_value, [$(32768)])
    tl.store($(vsort_1_value_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(32768))
    r10_lane = tl.arange($(32768))
    r10_row_values = tl.load($(vto_copy_7_ptr) + (r10_lane[None, :] % $(32768)), dtype=tl.int64)
    r10_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r10_other in range($(32768)) {
      r10_rank = r10_rank + tl.where(((tl.load($(vto_copy_7_ptr) + (r10_other % $(32768)), dtype=tl.int64) > r10_row_values) | ((tl.load($(vto_copy_7_ptr) + (r10_other % $(32768)), dtype=tl.int64) == r10_row_values) & (r10_other < r10_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r10_selected = r10_rank == tl.cast(r10_z[:, None], tl.int64)
    r10_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r10_cutoff in range($(32768)) {
      r10_value = r10_value + tl.cast(tl.sum(r10_selected & (r10_lane[None, :] > r10_cutoff), axis=1), tl.int64)
    }
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(vsort_1_index_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(524288))
    r11_value = tl.load($(input0_ptr) + ((((r11_z / $(2048)) % $(256)) * $(2048)) + (r11_z % $(2048))), dtype=tl.float64)
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(524288)])
    tl.store($(vto_copy_8_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(524288))
    r12_value = tl.load($(vto_copy_8_ptr) + ((((r12_z / $(2048)) % $(256)) * $(2048)) + (r12_z % $(2048))), dtype=tl.bfloat16)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(524288)])
    tl.store($(vto_copy_9_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(3145728))
    r13_value = tl.load($(vto_copy_9_ptr) + (((tl.load($(vremainder_ptr) + ((r13_z / $(2048)) % $(1536)), dtype=tl.int64) % $(256)) * $(2048)) + (r13_z % $(2048))), dtype=tl.float32)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(3145728)])
    tl.store($(vgather_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(3145728))
    r14_value = (tl.load($(vgather_ptr) + ((((r14_z / $(2048)) % $(1536)) * $(2048)) + (r14_z % $(2048))), dtype=tl.float32) * tl.load($(vto_copy_6_ptr) + ((((r14_z / $(2048)) % $(1536)) * $(2048)) + (r14_z % $(2048))), dtype=tl.float32))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(3145728)])
    tl.store($(vmul_1_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(1536))
    r15_lane = tl.arange($(2048))
    r15_value = tl.sum(tl.load($(vmul_1_ptr) + ((r15_z[:, None] * $(2048)) + r15_lane[None, :]), dtype=tl.float32), axis=1)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(1536)])
    tl.store($(vsum_3_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(32768))
    r16_value = 0.0
    for r16_lane in range($(1536)) {
      r16_value = tl.where(tl.load($(vsort_1_index_ptr) + (r16_lane % $(32768)), dtype=tl.int64) == r16_z, r16_value + tl.load($(vsum_3_ptr) + r16_lane, dtype=tl.float32), r16_value)
    }
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(32768)])
    tl.store($(vindex_put_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_value = tl.load($(vindex_put_ptr) + (((r17_z % $(128)) * $(256)) + ((r17_z / $(128)) % $(256))), dtype=tl.float32)
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(vto_copy_10_ptr) + r17_z, r17_value)
    z = tl.arange($(32768))
    value = tl.load($(vto_copy_10_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vtopk_value_ptr : RegionName := "vtopk_value"
  let vtopk_index_ptr : RegionName := "vtopk_index"
  let vscatter_ptr : RegionName := "vscatter"
  let vto_copy_8_ptr : RegionName := "vto_copy_8"
  let vto_copy_9_ptr : RegionName := "vto_copy_9"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vto_copy_10_ptr : RegionName := "vto_copy_10"
  let candidate_vtopk_value_ptr : RegionName := "candidate_vtopk_value"
  let candidate_vtopk_index_ptr : RegionName := "candidate_vtopk_index"
  let candidate_vscatter_ptr : RegionName := "candidate_vscatter"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vcumsum_ptr : RegionName := "candidate_vcumsum"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
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
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vwhere_3_ptr : RegionName := "candidate_vwhere_3"
  let candidate_vto_copy_8_ptr : RegionName := "candidate_vto_copy_8"
  let candidate_vto_copy_9_ptr : RegionName := "candidate_vto_copy_9"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vto_copy_10_ptr : RegionName := "candidate_vto_copy_10"
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
    r3_z = tl.arange($(32768))
    r3_value = tl.where(tl.load($(candidate_vscatter_ptr) + ((((r3_z / $(128)) % $(256)) * $(128)) + (r3_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(32768)])
    tl.store($(candidate_vto_copy_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = (tl.load($(candidate_vto_copy_1_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(candidate_vto_copy_2_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_value = tl.where(tl.load($(candidate_vto_copy_2_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (r5_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(candidate_vto_copy_3_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(128))
    r6_value = tl.full([$(128)], $(0), dtype=tl.int64)
    for r6_lane in range($(256)) {
      r6_value = r6_value + tl.load($(candidate_vto_copy_3_ptr) + ((r6_lane * $(128)) + r6_z), dtype=tl.int64)
    }
    r6_value = tl.broadcast_to(r6_value, [$(128)])
    tl.store($(candidate_vsum_2_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(128))
    r7_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(128)])
    for r7_lane in range($(128)) {
      r7_value = tl.where(r7_lane <= r7_z, (r7_value + tl.load($(candidate_vsum_2_ptr) + (r7_lane % $(128)), dtype=tl.int64)), r7_value)
    }
    r7_value = tl.broadcast_to(r7_value, [$(128)])
    tl.store($(candidate_vcumsum_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(128))
    r8_value = (tl.load($(candidate_vcumsum_ptr) + r8_z, dtype=tl.int64) - tl.load($(candidate_vsum_2_ptr) + r8_z, dtype=tl.int64))
    r8_value = tl.broadcast_to(r8_value, [$(128)])
    tl.store($(candidate_vsub_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(32768))
    r9_value = tl.where(tl.load($(candidate_vto_copy_2_ptr) + ((((r9_z / $(128)) % $(256)) * $(128)) + (r9_z % $(128))), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(32768)])
    tl.store($(candidate_vto_copy_4_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(32768))
    r10_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(32768)])
    for r10_lane in range($(256)) {
      r10_value = tl.where(r10_lane <= ((r10_z / $(128)) % $(256)), (r10_value + tl.load($(candidate_vto_copy_4_ptr) + (((r10_lane % $(256)) * $(128)) + (r10_z % $(128))), dtype=tl.int64)), r10_value)
    }
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(candidate_vcumsum_1_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(32768))
    r11_value = (tl.load($(candidate_vcumsum_1_ptr) + ((((r11_z / $(128)) % $(256)) * $(128)) + (r11_z % $(128))), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r11_value = tl.broadcast_to(r11_value, [$(32768)])
    tl.store($(candidate_vsub_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(32768))
    r12_value = (tl.load($(candidate_vsub_1_ptr) + ((((r12_z / $(128)) % $(256)) * $(128)) + (r12_z % $(128))), dtype=tl.int64) + tl.load($(candidate_vsub_ptr) + (r12_z % $(128)), dtype=tl.int64))
    r12_value = tl.broadcast_to(r12_value, [$(32768)])
    tl.store($(candidate_vadd_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(32768))
    r13_value = tl.where(tl.load($(candidate_vto_copy_2_ptr) + ((((r13_z / $(128)) % $(256)) * $(128)) + (r13_z % $(128))), dtype=tl.int1), tl.load($(candidate_vadd_ptr) + ((((r13_z / $(128)) % $(256)) * $(128)) + (r13_z % $(128))), dtype=tl.int64), (-tl.full([], $(1), dtype=tl.int64)))
    r13_value = tl.broadcast_to(r13_value, [$(32768)])
    tl.store($(candidate_vwhere_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(32768))
    r14_value = tl.where(tl.load($(candidate_vto_copy_2_ptr) + ((((r14_z / $(128)) % $(256)) * $(128)) + (r14_z % $(128))), dtype=tl.int1), tl.load($(candidate_vwhere_ptr) + ((((r14_z / $(128)) % $(256)) * $(128)) + (r14_z % $(128))), dtype=tl.int64), tl.full([], $(1536), dtype=tl.int64))
    r14_value = tl.broadcast_to(r14_value, [$(32768)])
    tl.store($(candidate_vwhere_1_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(67108864))
    r15_value = tl.where(tl.load($(candidate_vto_copy_2_ptr) + ((((r15_z / $(262144)) % $(256)) * $(128)) + ((r15_z / $(2048)) % $(128))), dtype=tl.int1), tl.load($(parameter1_ptr) + ((((r15_z / $(262144)) % $(256)) * $(2048)) + (r15_z % $(2048))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(67108864)])
    tl.store($(candidate_vwhere_2_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(3147776))
    r16_value = tl.broadcast_to(tl.cast(0.0, tl.bfloat16), [$(3147776)])
    for r16_lane in range($(32768)) {
      r16_value = tl.where(tl.load($(candidate_vwhere_1_ptr) + ((((r16_lane / $(128)) % $(256)) * $(128)) + (r16_lane % $(128))), dtype=tl.int64) == tl.cast(((r16_z / $(2048)) % $(1537)), tl.int64), (r16_value + tl.load($(candidate_vwhere_2_ptr) + ((((((r16_lane * $(2048)) + (r16_z % $(2048))) / $(262144)) % $(256)) * $(262144)) + ((r16_lane % $(128)) * $(2048)) + (r16_z % $(2048))), dtype=tl.bfloat16)), r16_value)
    }
    r16_value = r16_value.to(tl.bfloat16)
    r16_value = tl.broadcast_to(r16_value, [$(3147776)])
    tl.store($(candidate_vscatter_add_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_value = (tl.load($(candidate_vwhere_ptr) + ((((r17_z / $(128)) % $(256)) * $(128)) + (r17_z % $(128))), dtype=tl.int64) >= tl.full([], $(0), dtype=tl.int64))
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(candidate_vge_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = tl.maximum(tl.load($(candidate_vwhere_ptr) + ((((r18_z / $(128)) % $(256)) * $(128)) + (r18_z % $(128))), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(candidate_vclamp_min_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(67108864))
    r19_value = tl.load($(candidate_vscatter_add_ptr) + (((tl.load($(candidate_vclamp_min_ptr) + ((((((r19_z / $(2048)) % $(32768)) / $(128)) % $(256)) * $(128)) + (((r19_z / $(2048)) % $(32768)) % $(128))), dtype=tl.int64) % $(1537)) * $(2048)) + (r19_z % $(2048))), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.bfloat16)
    r19_value = tl.broadcast_to(r19_value, [$(67108864)])
    tl.store($(candidate_vindex_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(67108864))
    r20_value = tl.load($(candidate_vindex_ptr) + ((((r20_z / $(262144)) % $(256)) * $(262144)) + (((r20_z / $(2048)) % $(128)) * $(2048)) + (r20_z % $(2048))), dtype=tl.bfloat16)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(67108864)])
    tl.store($(candidate_vto_copy_5_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(67108864))
    r21_value = tl.where(tl.load($(candidate_vge_ptr) + ((((r21_z / $(262144)) % $(256)) * $(128)) + ((r21_z / $(2048)) % $(128))), dtype=tl.int1), tl.load($(candidate_vto_copy_5_ptr) + ((((r21_z / $(262144)) % $(256)) * $(262144)) + (((r21_z / $(2048)) % $(128)) * $(2048)) + (r21_z % $(2048))), dtype=tl.float32), 0.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(67108864)])
    tl.store($(candidate_vwhere_3_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(524288))
    r22_value = tl.load($(input0_ptr) + ((((r22_z / $(2048)) % $(256)) * $(2048)) + (r22_z % $(2048))), dtype=tl.float64)
    r22_value = r22_value.to(tl.bfloat16)
    r22_value = tl.broadcast_to(r22_value, [$(524288)])
    tl.store($(candidate_vto_copy_8_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(524288))
    r23_value = tl.load($(candidate_vto_copy_8_ptr) + ((((r23_z / $(2048)) % $(256)) * $(2048)) + (r23_z % $(2048))), dtype=tl.bfloat16)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(524288)])
    tl.store($(candidate_vto_copy_9_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(67108864))
    r24_value = (tl.load($(candidate_vto_copy_9_ptr) + ((((r24_z / $(262144)) % $(256)) * $(2048)) + (r24_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vwhere_3_ptr) + ((((r24_z / $(262144)) % $(256)) * $(262144)) + (((r24_z / $(2048)) % $(128)) * $(2048)) + (r24_z % $(2048))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(67108864)])
    tl.store($(candidate_vmul_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(32768))
    r25_lane = tl.arange($(2048))
    r25_value = tl.sum(tl.load($(candidate_vmul_1_ptr) + ((((r25_z[:, None] / $(128)) % $(256)) * $(262144)) + ((r25_z[:, None] % $(128)) * $(2048)) + r25_lane[None, :]), dtype=tl.float32), axis=1)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(32768)])
    tl.store($(candidate_vsum_4_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(32768))
    r26_value = tl.load($(candidate_vsum_4_ptr) + ((((r26_z / $(128)) % $(256)) * $(128)) + (r26_z % $(128))), dtype=tl.float32)
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(32768)])
    tl.store($(candidate_vto_copy_10_ptr) + r26_z, r26_value)
    z = tl.arange($(32768))
    value = tl.load($(candidate_vto_copy_10_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_unpermute_dmerging_probs
