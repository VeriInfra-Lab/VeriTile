import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_permute_with_probs
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
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vscatter_1_ptr : RegionName := "candidate_vscatter_1"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vsort_value_ptr : RegionName := "candidate_vsort_value"
  let candidate_vsort_index_ptr : RegionName := "candidate_vsort_index"
  let candidate_vremainder_ptr : RegionName := "candidate_vremainder"
  let candidate_vindex_ptr : RegionName := "candidate_vindex"
  let candidate_vindex_select_ptr : RegionName := "candidate_vindex_select"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vscatter_add_ptr : RegionName := "candidate_vscatter_add"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
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
    r3_lane = tl.arange($(6))
    r3_row_values = tl.load($(candidate_vtopk_value_ptr) + ((((r3_z[:, None] / $(6)) % $(256)) * $(6)) + (r3_lane[None, :] % $(6))), dtype=tl.float32)
    r3_maximum = tl.max(r3_row_values, axis=1, keep_dims=true)
    r3_weights = tl.exp(r3_row_values - r3_maximum)
    r3_total = tl.sum(r3_weights, axis=1)
    r3_maximum = tl.max(r3_row_values, axis=1)
    r3_value = tl.exp(tl.load($(candidate_vtopk_value_ptr) + ((((r3_z / $(6)) % $(256)) * $(6)) + (r3_z % $(6))), dtype=tl.float32) - r3_maximum) / r3_total
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(1536)])
    tl.store($(candidate_vsoftmax_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = tl.broadcast_to(0.0, [$(32768)])
    for r4_lane in range($(6)) {
      r4_value = tl.where(tl.load($(candidate_vtopk_index_ptr) + ((((r4_z / $(128)) % $(256)) * $(6)) + (r4_lane % $(6))), dtype=tl.int64) == tl.cast((r4_z % $(128)), tl.int64), tl.load($(candidate_vsoftmax_ptr) + ((((r4_z / $(128)) % $(256)) * $(6)) + (r4_lane % $(6))), dtype=tl.float32), r4_value)
    }
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(candidate_vscatter_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_value = tl.load($(candidate_vscatter_1_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (r5_z % $(128))), dtype=tl.float32)
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(candidate_vto_copy_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(524288))
    r6_value = tl.load($(input0_ptr) + ((((r6_z / $(2048)) % $(256)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float64)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(524288)])
    tl.store($(candidate_vto_copy_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_lane = tl.arange($(32768))
    r7_row_values = tl.load($(candidate_vscatter_ptr) + (((r7_lane[None, :] % $(256)) * $(128)) + ((r7_lane[None, :] / $(256)) % $(128))), dtype=tl.int1)
    r7_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r7_other in range($(32768)) {
      r7_rank = r7_rank + tl.where(((tl.load($(candidate_vscatter_ptr) + (((r7_other % $(256)) * $(128)) + ((r7_other / $(256)) % $(128))), dtype=tl.int1) & (~r7_row_values)) | (((tl.load($(candidate_vscatter_ptr) + (((r7_other % $(256)) * $(128)) + ((r7_other / $(256)) % $(128))), dtype=tl.int1) & r7_row_values) | ((~tl.load($(candidate_vscatter_ptr) + (((r7_other % $(256)) * $(128)) + ((r7_other / $(256)) % $(128))), dtype=tl.int1)) & (~r7_row_values))) & (r7_other < r7_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r7_selected = r7_rank == tl.cast(r7_z[:, None], tl.int64)
    r7_value = tl.sum(r7_selected & r7_row_values, axis=1) > $(0)
    r7_value = tl.broadcast_to(r7_value, [$(32768)])
    tl.store($(candidate_vsort_value_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32768))
    r8_lane = tl.arange($(32768))
    r8_row_values = tl.load($(candidate_vscatter_ptr) + (((r8_lane[None, :] % $(256)) * $(128)) + ((r8_lane[None, :] / $(256)) % $(128))), dtype=tl.int1)
    r8_rank = tl.full([$(32768), $(32768)], $(0), dtype=tl.int64)
    for r8_other in range($(32768)) {
      r8_rank = r8_rank + tl.where(((tl.load($(candidate_vscatter_ptr) + (((r8_other % $(256)) * $(128)) + ((r8_other / $(256)) % $(128))), dtype=tl.int1) & (~r8_row_values)) | (((tl.load($(candidate_vscatter_ptr) + (((r8_other % $(256)) * $(128)) + ((r8_other / $(256)) % $(128))), dtype=tl.int1) & r8_row_values) | ((~tl.load($(candidate_vscatter_ptr) + (((r8_other % $(256)) * $(128)) + ((r8_other / $(256)) % $(128))), dtype=tl.int1)) & (~r8_row_values))) & (r8_other < r8_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r8_selected = r8_rank == tl.cast(r8_z[:, None], tl.int64)
    r8_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r8_cutoff in range($(32768)) {
      r8_value = r8_value + tl.cast(tl.sum(r8_selected & (r8_lane[None, :] > r8_cutoff), axis=1), tl.int64)
    }
    r8_value = tl.broadcast_to(r8_value, [$(32768)])
    tl.store($(candidate_vsort_index_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(1536))
    r9_value = (tl.load($(candidate_vsort_index_ptr) + r9_z, dtype=tl.int64) % tl.full([], $(256), dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(1536)])
    tl.store($(candidate_vremainder_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(1536))
    r10_value = tl.load($(candidate_vto_copy_ptr) + ((((tl.load($(candidate_vsort_index_ptr) + r10_z, dtype=tl.int64) % $(256)) % $(256)) * $(128)) + (((tl.load($(candidate_vsort_index_ptr) + r10_z, dtype=tl.int64) / $(256)) % $(128)) % $(128))), dtype=tl.bfloat16)
    r10_value = r10_value.to(tl.bfloat16)
    r10_value = tl.broadcast_to(r10_value, [$(1536)])
    tl.store($(candidate_vindex_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(3145728))
    r11_value = tl.load($(candidate_vto_copy_1_ptr) + (((tl.load($(candidate_vremainder_ptr) + ((r11_z / $(2048)) % $(1536)), dtype=tl.int64) % $(256)) * $(2048)) + (r11_z % $(2048))), dtype=tl.bfloat16)
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(3145728)])
    tl.store($(candidate_vindex_select_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(3145728))
    r12_value = tl.load($(candidate_vindex_select_ptr) + ((((r12_z / $(2048)) % $(1536)) * $(2048)) + (r12_z % $(2048))), dtype=tl.bfloat16)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(3145728)])
    tl.store($(candidate_vto_copy_2_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(1536))
    r13_value = tl.load($(candidate_vindex_ptr) + r13_z, dtype=tl.bfloat16)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(1536)])
    tl.store($(candidate_vto_copy_3_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(3145728))
    r14_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r14_z / $(2048)) % $(1536)) * $(2048)) + (r14_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vto_copy_3_ptr) + ((r14_z / $(2048)) % $(1536)), dtype=tl.float32))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(3145728)])
    tl.store($(candidate_vmul_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(524288))
    r15_value = tl.broadcast_to(0.0, [$(524288)])
    for r15_lane in range($(1536)) {
      r15_value = tl.where(tl.load($(candidate_vremainder_ptr) + r15_lane, dtype=tl.int64) == tl.cast(((r15_z / $(2048)) % $(256)), tl.int64), (r15_value + tl.load($(candidate_vmul_ptr) + (((r15_lane % $(1536)) * $(2048)) + (r15_z % $(2048))), dtype=tl.float32)), r15_value)
    }
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(524288)])
    tl.store($(candidate_vscatter_add_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(524288))
    r16_value = tl.load($(candidate_vscatter_add_ptr) + ((((r16_z / $(2048)) % $(256)) * $(2048)) + (r16_z % $(2048))), dtype=tl.float32)
    r16_value = r16_value.to(tl.bfloat16)
    r16_value = tl.broadcast_to(r16_value, [$(524288)])
    tl.store($(candidate_vto_copy_4_ptr) + r16_z, r16_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vto_copy_4_ptr) + ((((z / $(2048)) % $(256)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_permute_with_probs
