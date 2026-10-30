import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_router_sigmoid_bias_optimized
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
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vremainder_ptr : RegionName := "vremainder"
  let vindex_put_ptr : RegionName := "vindex_put"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vadd_1_ptr : RegionName := "vadd_1"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vgather_1_ptr : RegionName := "vgather_1"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vscatter_ptr : RegionName := "vscatter"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(16))
    r1_value = (tl.cast(r1_z, tl.int64) * tl.full([], $(17), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(16)])
    tl.store($(vmul_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(16))
    r2_value = (tl.load($(vmul_ptr) + r2_z, dtype=tl.int64) + tl.full([], $(7), dtype=tl.int64))
    r2_value = tl.broadcast_to(r2_value, [$(16)])
    tl.store($(vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(16))
    r3_value = (tl.load($(vadd_ptr) + r3_z, dtype=tl.int64) % tl.full([], $(128), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(16)])
    tl.store($(vremainder_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(128))
    r4_value = 0.0
    for r4_lane in range($(16)) {
      r4_value = tl.where(tl.load($(vremainder_ptr) + (r4_lane % $(16)), dtype=tl.int64) == r4_z, (0.05 + r4_lane * 0.002), r4_value)
    }
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(128)])
    tl.store($(vindex_put_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2176))
    r5_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r5_z / $(128)) % $(17)) * $(128)) + (r5_z % $(128))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(2176)])
    tl.store($(vsigmoid_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2176))
    r6_value = (tl.load($(vsigmoid_ptr) + ((((r6_z / $(128)) % $(17)) * $(128)) + (r6_z % $(128))), dtype=tl.float32) + tl.load($(vindex_put_ptr) + (r6_z % $(128)), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(2176)])
    tl.store($(vadd_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(2176))
    r7_value = (tl.load($(vadd_1_ptr) + ((((r7_z / $(128)) % $(17)) * $(128)) + (r7_z % $(128))), dtype=tl.float32) != tl.load($(vadd_1_ptr) + ((((r7_z / $(128)) % $(17)) * $(128)) + (r7_z % $(128))), dtype=tl.float32))
    r7_value = tl.broadcast_to(r7_value, [$(2176)])
    tl.store($(visnan_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(2176))
    r8_value = tl.where(tl.load($(visnan_ptr) + ((((r8_z / $(128)) % $(17)) * $(128)) + (r8_z % $(128))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(vadd_1_ptr) + ((((r8_z / $(128)) % $(17)) * $(128)) + (r8_z % $(128))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(2176)])
    tl.store($(vwhere_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(2176))
    r9_lane = tl.arange($(128))
    r9_row_values = tl.load($(vwhere_ptr) + ((((r9_z[:, None] / $(128)) % $(17)) * $(128)) + (r9_lane[None, :] % $(128))), dtype=tl.float32)
    r9_rank = tl.full([$(2176), $(128)], $(0), dtype=tl.int64)
    for r9_other in range($(128)) {
      r9_other_value = tl.load($(vwhere_ptr) + ((((r9_z[:, None] / $(128)) % $(17)) * $(128)) + (r9_other % $(128))), dtype=tl.float32)
      r9_other_nan = r9_other_value != r9_other_value
      r9_row_nan = r9_row_values != r9_row_values
      r9_strict = (r9_other_nan & (~r9_row_nan)) | (((~r9_other_nan) & (~r9_row_nan)) & (r9_other_value > r9_row_values))
      r9_equal = (r9_other_nan & r9_row_nan) | (r9_other_value == r9_row_values)
      r9_rank = r9_rank + tl.where(r9_strict | (r9_equal & (r9_other < r9_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r9_selected = r9_rank == tl.cast((r9_z % $(128))[:, None], tl.int64)
    r9_value = tl.full([$(2176)], $(0), dtype=tl.int64)
    for r9_cutoff in range($(128)) {
      r9_value = r9_value + tl.cast(tl.sum(r9_selected & (r9_lane[None, :] > r9_cutoff), axis=1), tl.int64)
    }
    r9_value = tl.load($(vwhere_ptr) + ((((r9_z / $(128)) % $(17)) * $(128)) + (tl.cast(r9_value, tl.uint32) % $(128))), dtype=tl.float32)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(2176)])
    tl.store($(vsort_value_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(2176))
    r10_lane = tl.arange($(128))
    r10_row_values = tl.load($(vwhere_ptr) + ((((r10_z[:, None] / $(128)) % $(17)) * $(128)) + (r10_lane[None, :] % $(128))), dtype=tl.float32)
    r10_rank = tl.full([$(2176), $(128)], $(0), dtype=tl.int64)
    for r10_other in range($(128)) {
      r10_other_value = tl.load($(vwhere_ptr) + ((((r10_z[:, None] / $(128)) % $(17)) * $(128)) + (r10_other % $(128))), dtype=tl.float32)
      r10_other_nan = r10_other_value != r10_other_value
      r10_row_nan = r10_row_values != r10_row_values
      r10_strict = (r10_other_nan & (~r10_row_nan)) | (((~r10_other_nan) & (~r10_row_nan)) & (r10_other_value > r10_row_values))
      r10_equal = (r10_other_nan & r10_row_nan) | (r10_other_value == r10_row_values)
      r10_rank = r10_rank + tl.where(r10_strict | (r10_equal & (r10_other < r10_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r10_selected = r10_rank == tl.cast((r10_z % $(128))[:, None], tl.int64)
    r10_value = tl.full([$(2176)], $(0), dtype=tl.int64)
    for r10_cutoff in range($(128)) {
      r10_value = r10_value + tl.cast(tl.sum(r10_selected & (r10_lane[None, :] > r10_cutoff), axis=1), tl.int64)
    }
    r10_value = tl.broadcast_to(r10_value, [$(2176)])
    tl.store($(vsort_index_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(272))
    r11_value = tl.load($(vsigmoid_ptr) + ((((r11_z / $(16)) % $(17)) * $(128)) + (tl.load($(vsort_index_ptr) + ((((r11_z / $(16)) % $(17)) * $(128)) + (r11_z % $(16))), dtype=tl.int64) % $(128))), dtype=tl.float32)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(272)])
    tl.store($(vgather_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(17))
    r12_lane = tl.arange($(16))
    r12_value = tl.sum(tl.load($(vgather_1_ptr) + ((r12_z[:, None] * $(16)) + r12_lane[None, :]), dtype=tl.float32), axis=1)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(17)])
    tl.store($(vsum_1_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(17))
    r13_value = (tl.load($(vsum_1_ptr) + r13_z, dtype=tl.float32) + 0.00000000000000000001)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(17)])
    tl.store($(vadd_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(272))
    r14_value = (tl.load($(vgather_1_ptr) + ((((r14_z / $(16)) % $(17)) * $(16)) + (r14_z % $(16))), dtype=tl.float32) / tl.load($(vadd_2_ptr) + ((r14_z / $(16)) % $(17)), dtype=tl.float32))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(272)])
    tl.store($(vdiv_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(272))
    r15_value = (tl.load($(vdiv_ptr) + ((((r15_z / $(16)) % $(17)) * $(16)) + (r15_z % $(16))), dtype=tl.float32) * 2.826)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(272)])
    tl.store($(vmul_1_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(2176))
    r16_value = tl.broadcast_to(0.0, [$(2176)])
    for r16_lane in range($(16)) {
      r16_value = tl.where(tl.load($(vsort_index_ptr) + ((((r16_z / $(128)) % $(17)) * $(128)) + (r16_lane % $(128))), dtype=tl.int64) == tl.cast((r16_z % $(128)), tl.int64), tl.load($(vmul_1_ptr) + ((((r16_z / $(128)) % $(17)) * $(16)) + (r16_lane % $(16))), dtype=tl.float32), r16_value)
    }
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(2176)])
    tl.store($(vscatter_ptr) + r16_z, r16_value)
    z = tl.arange($(2176))
    value = tl.load($(vscatter_ptr) + ((((z / $(128)) % $(17)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(2176)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vremainder_ptr : RegionName := "vremainder"
  let vindex_put_ptr : RegionName := "vindex_put"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vscatter_ptr : RegionName := "vscatter"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vremainder_ptr : RegionName := "candidate_vremainder"
  let candidate_vindex_put_ptr : RegionName := "candidate_vindex_put"
  let candidate_vsigmoid_ptr : RegionName := "candidate_vsigmoid"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vsort_value_ptr : RegionName := "candidate_vsort_value"
  let candidate_vsort_index_ptr : RegionName := "candidate_vsort_index"
  let candidate_vgather_ptr : RegionName := "candidate_vgather"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vscatter_ptr : RegionName := "candidate_vscatter"
  triton {
    r0_z = tl.arange($(16))
    r0_value = (tl.cast(r0_z, tl.int64) * tl.full([], $(17), dtype=tl.int64))
    r0_value = tl.broadcast_to(r0_value, [$(16)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(16))
    r1_value = (tl.load($(candidate_vmul_ptr) + r1_z, dtype=tl.int64) + tl.full([], $(7), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(16)])
    tl.store($(candidate_vadd_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(16))
    r2_value = (tl.load($(candidate_vadd_ptr) + r2_z, dtype=tl.int64) % tl.full([], $(128), dtype=tl.int64))
    r2_value = tl.broadcast_to(r2_value, [$(16)])
    tl.store($(candidate_vremainder_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(128))
    r3_value = 0.0
    for r3_lane in range($(16)) {
      r3_value = tl.where(tl.load($(candidate_vremainder_ptr) + (r3_lane % $(16)), dtype=tl.int64) == r3_z, (0.05 + r3_lane * 0.002), r3_value)
    }
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(128)])
    tl.store($(candidate_vindex_put_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(2176))
    r4_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r4_z / $(128)) % $(17)) * $(128)) + (r4_z % $(128))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(2176)])
    tl.store($(candidate_vsigmoid_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2176))
    r5_value = (tl.load($(candidate_vsigmoid_ptr) + ((((r5_z / $(128)) % $(17)) * $(128)) + (r5_z % $(128))), dtype=tl.float32) + tl.load($(candidate_vindex_put_ptr) + (r5_z % $(128)), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(2176)])
    tl.store($(candidate_vadd_1_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2176))
    r6_lane = tl.arange($(128))
    r6_row_values = tl.load($(candidate_vadd_1_ptr) + ((((r6_z[:, None] / $(128)) % $(17)) * $(128)) + (r6_lane[None, :] % $(128))), dtype=tl.float32)
    r6_rank = tl.full([$(2176), $(128)], $(0), dtype=tl.int64)
    for r6_other in range($(128)) {
      r6_other_value = tl.load($(candidate_vadd_1_ptr) + ((((r6_z[:, None] / $(128)) % $(17)) * $(128)) + (r6_other % $(128))), dtype=tl.float32)
      r6_other_nan = r6_other_value != r6_other_value
      r6_row_nan = r6_row_values != r6_row_values
      r6_strict = (r6_other_nan & (~r6_row_nan)) | (((~r6_other_nan) & (~r6_row_nan)) & (r6_other_value > r6_row_values))
      r6_equal = (r6_other_nan & r6_row_nan) | (r6_other_value == r6_row_values)
      r6_rank = r6_rank + tl.where(r6_strict | (r6_equal & (r6_other < r6_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r6_selected = r6_rank == tl.cast((r6_z % $(128))[:, None], tl.int64)
    r6_value = tl.full([$(2176)], $(0), dtype=tl.int64)
    for r6_cutoff in range($(128)) {
      r6_value = r6_value + tl.cast(tl.sum(r6_selected & (r6_lane[None, :] > r6_cutoff), axis=1), tl.int64)
    }
    r6_value = tl.load($(candidate_vadd_1_ptr) + ((((r6_z / $(128)) % $(17)) * $(128)) + (tl.cast(r6_value, tl.uint32) % $(128))), dtype=tl.float32)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(2176)])
    tl.store($(candidate_vsort_value_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(2176))
    r7_lane = tl.arange($(128))
    r7_row_values = tl.load($(candidate_vadd_1_ptr) + ((((r7_z[:, None] / $(128)) % $(17)) * $(128)) + (r7_lane[None, :] % $(128))), dtype=tl.float32)
    r7_rank = tl.full([$(2176), $(128)], $(0), dtype=tl.int64)
    for r7_other in range($(128)) {
      r7_other_value = tl.load($(candidate_vadd_1_ptr) + ((((r7_z[:, None] / $(128)) % $(17)) * $(128)) + (r7_other % $(128))), dtype=tl.float32)
      r7_other_nan = r7_other_value != r7_other_value
      r7_row_nan = r7_row_values != r7_row_values
      r7_strict = (r7_other_nan & (~r7_row_nan)) | (((~r7_other_nan) & (~r7_row_nan)) & (r7_other_value > r7_row_values))
      r7_equal = (r7_other_nan & r7_row_nan) | (r7_other_value == r7_row_values)
      r7_rank = r7_rank + tl.where(r7_strict | (r7_equal & (r7_other < r7_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r7_selected = r7_rank == tl.cast((r7_z % $(128))[:, None], tl.int64)
    r7_value = tl.full([$(2176)], $(0), dtype=tl.int64)
    for r7_cutoff in range($(128)) {
      r7_value = r7_value + tl.cast(tl.sum(r7_selected & (r7_lane[None, :] > r7_cutoff), axis=1), tl.int64)
    }
    r7_value = tl.broadcast_to(r7_value, [$(2176)])
    tl.store($(candidate_vsort_index_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(272))
    r8_value = tl.load($(candidate_vsigmoid_ptr) + ((((r8_z / $(16)) % $(17)) * $(128)) + (tl.load($(candidate_vsort_index_ptr) + ((((r8_z / $(16)) % $(17)) * $(128)) + (r8_z % $(16))), dtype=tl.int64) % $(128))), dtype=tl.float32)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(272)])
    tl.store($(candidate_vgather_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(17))
    r9_lane = tl.arange($(16))
    r9_value = tl.sum(tl.load($(candidate_vgather_ptr) + ((r9_z[:, None] * $(16)) + r9_lane[None, :]), dtype=tl.float32), axis=1)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(17)])
    tl.store($(candidate_vsum_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(17))
    r10_value = (tl.load($(candidate_vsum_1_ptr) + r10_z, dtype=tl.float32) + 0.00000000000000000001)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(17)])
    tl.store($(candidate_vadd_2_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(272))
    r11_value = (tl.load($(candidate_vgather_ptr) + ((((r11_z / $(16)) % $(17)) * $(16)) + (r11_z % $(16))), dtype=tl.float32) / tl.load($(candidate_vadd_2_ptr) + ((r11_z / $(16)) % $(17)), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(272)])
    tl.store($(candidate_vdiv_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(272))
    r12_value = (tl.load($(candidate_vdiv_ptr) + ((((r12_z / $(16)) % $(17)) * $(16)) + (r12_z % $(16))), dtype=tl.float32) * 2.826)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(272)])
    tl.store($(candidate_vmul_1_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(2176))
    r13_value = tl.broadcast_to(0.0, [$(2176)])
    for r13_lane in range($(16)) {
      r13_value = tl.where(tl.load($(candidate_vsort_index_ptr) + ((((r13_z / $(128)) % $(17)) * $(128)) + (r13_lane % $(128))), dtype=tl.int64) == tl.cast((r13_z % $(128)), tl.int64), tl.load($(candidate_vmul_1_ptr) + ((((r13_z / $(128)) % $(17)) * $(16)) + (r13_lane % $(16))), dtype=tl.float32), r13_value)
    }
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(2176)])
    tl.store($(candidate_vscatter_ptr) + r13_z, r13_value)
    z = tl.arange($(2176))
    value = tl.load($(candidate_vscatter_ptr) + ((((z / $(128)) % $(17)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(2176)])
    tl.store($(out_ptr) + z, value)
  }

/-- Finite Real specialization. The non-real NaN literal is assigned the unused
Real payload zero. Its connection to `original` is a separate public goal. -/
def realOriginal : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan_ptr : RegionName := "literal_nan"
  let vmul_ptr : RegionName := "vmul"
  let vadd_ptr : RegionName := "vadd"
  let vremainder_ptr : RegionName := "vremainder"
  let vindex_put_ptr : RegionName := "vindex_put"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vadd_1_ptr : RegionName := "vadd_1"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vgather_1_ptr : RegionName := "vgather_1"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vscatter_ptr : RegionName := "vscatter"
  triton {
    r0_z = tl.arange($(1))
    r0_value = 0.0
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(16))
    r1_value = (tl.cast(r1_z, tl.int64) * tl.full([], $(17), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(16)])
    tl.store($(vmul_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(16))
    r2_value = (tl.load($(vmul_ptr) + r2_z, dtype=tl.int64) + tl.full([], $(7), dtype=tl.int64))
    r2_value = tl.broadcast_to(r2_value, [$(16)])
    tl.store($(vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(16))
    r3_value = (tl.load($(vadd_ptr) + r3_z, dtype=tl.int64) % tl.full([], $(128), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(16)])
    tl.store($(vremainder_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(128))
    r4_value = 0.0
    for r4_lane in range($(16)) {
      r4_value = tl.where(tl.load($(vremainder_ptr) + (r4_lane % $(16)), dtype=tl.int64) == r4_z, (0.05 + r4_lane * 0.002), r4_value)
    }
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(128)])
    tl.store($(vindex_put_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2176))
    r5_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r5_z / $(128)) % $(17)) * $(128)) + (r5_z % $(128))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(2176)])
    tl.store($(vsigmoid_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2176))
    r6_value = (tl.load($(vsigmoid_ptr) + ((((r6_z / $(128)) % $(17)) * $(128)) + (r6_z % $(128))), dtype=tl.float32) + tl.load($(vindex_put_ptr) + (r6_z % $(128)), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(2176)])
    tl.store($(vadd_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(2176))
    r7_value = (tl.load($(vadd_1_ptr) + ((((r7_z / $(128)) % $(17)) * $(128)) + (r7_z % $(128))), dtype=tl.float32) != tl.load($(vadd_1_ptr) + ((((r7_z / $(128)) % $(17)) * $(128)) + (r7_z % $(128))), dtype=tl.float32))
    r7_value = tl.broadcast_to(r7_value, [$(2176)])
    tl.store($(visnan_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(2176))
    r8_value = tl.where(tl.load($(visnan_ptr) + ((((r8_z / $(128)) % $(17)) * $(128)) + (r8_z % $(128))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(vadd_1_ptr) + ((((r8_z / $(128)) % $(17)) * $(128)) + (r8_z % $(128))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(2176)])
    tl.store($(vwhere_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(2176))
    r9_lane = tl.arange($(128))
    r9_row_values = tl.load($(vwhere_ptr) + ((((r9_z[:, None] / $(128)) % $(17)) * $(128)) + (r9_lane[None, :] % $(128))), dtype=tl.float32)
    r9_rank = tl.full([$(2176), $(128)], $(0), dtype=tl.int64)
    for r9_other in range($(128)) {
      r9_other_value = tl.load($(vwhere_ptr) + ((((r9_z[:, None] / $(128)) % $(17)) * $(128)) + (r9_other % $(128))), dtype=tl.float32)
      r9_other_nan = r9_other_value != r9_other_value
      r9_row_nan = r9_row_values != r9_row_values
      r9_strict = (r9_other_nan & (~r9_row_nan)) | (((~r9_other_nan) & (~r9_row_nan)) & (r9_other_value > r9_row_values))
      r9_equal = (r9_other_nan & r9_row_nan) | (r9_other_value == r9_row_values)
      r9_rank = r9_rank + tl.where(r9_strict | (r9_equal & (r9_other < r9_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r9_selected = r9_rank == tl.cast((r9_z % $(128))[:, None], tl.int64)
    r9_value = tl.full([$(2176)], $(0), dtype=tl.int64)
    for r9_cutoff in range($(128)) {
      r9_value = r9_value + tl.cast(tl.sum(r9_selected & (r9_lane[None, :] > r9_cutoff), axis=1), tl.int64)
    }
    r9_value = tl.load($(vwhere_ptr) + ((((r9_z / $(128)) % $(17)) * $(128)) + (tl.cast(r9_value, tl.uint32) % $(128))), dtype=tl.float32)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(2176)])
    tl.store($(vsort_value_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(2176))
    r10_lane = tl.arange($(128))
    r10_row_values = tl.load($(vwhere_ptr) + ((((r10_z[:, None] / $(128)) % $(17)) * $(128)) + (r10_lane[None, :] % $(128))), dtype=tl.float32)
    r10_rank = tl.full([$(2176), $(128)], $(0), dtype=tl.int64)
    for r10_other in range($(128)) {
      r10_other_value = tl.load($(vwhere_ptr) + ((((r10_z[:, None] / $(128)) % $(17)) * $(128)) + (r10_other % $(128))), dtype=tl.float32)
      r10_other_nan = r10_other_value != r10_other_value
      r10_row_nan = r10_row_values != r10_row_values
      r10_strict = (r10_other_nan & (~r10_row_nan)) | (((~r10_other_nan) & (~r10_row_nan)) & (r10_other_value > r10_row_values))
      r10_equal = (r10_other_nan & r10_row_nan) | (r10_other_value == r10_row_values)
      r10_rank = r10_rank + tl.where(r10_strict | (r10_equal & (r10_other < r10_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r10_selected = r10_rank == tl.cast((r10_z % $(128))[:, None], tl.int64)
    r10_value = tl.full([$(2176)], $(0), dtype=tl.int64)
    for r10_cutoff in range($(128)) {
      r10_value = r10_value + tl.cast(tl.sum(r10_selected & (r10_lane[None, :] > r10_cutoff), axis=1), tl.int64)
    }
    r10_value = tl.broadcast_to(r10_value, [$(2176)])
    tl.store($(vsort_index_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(272))
    r11_value = tl.load($(vsigmoid_ptr) + ((((r11_z / $(16)) % $(17)) * $(128)) + (tl.load($(vsort_index_ptr) + ((((r11_z / $(16)) % $(17)) * $(128)) + (r11_z % $(16))), dtype=tl.int64) % $(128))), dtype=tl.float32)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(272)])
    tl.store($(vgather_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(17))
    r12_lane = tl.arange($(16))
    r12_value = tl.sum(tl.load($(vgather_1_ptr) + ((r12_z[:, None] * $(16)) + r12_lane[None, :]), dtype=tl.float32), axis=1)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(17)])
    tl.store($(vsum_1_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(17))
    r13_value = (tl.load($(vsum_1_ptr) + r13_z, dtype=tl.float32) + 0.00000000000000000001)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(17)])
    tl.store($(vadd_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(272))
    r14_value = (tl.load($(vgather_1_ptr) + ((((r14_z / $(16)) % $(17)) * $(16)) + (r14_z % $(16))), dtype=tl.float32) / tl.load($(vadd_2_ptr) + ((r14_z / $(16)) % $(17)), dtype=tl.float32))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(272)])
    tl.store($(vdiv_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(272))
    r15_value = (tl.load($(vdiv_ptr) + ((((r15_z / $(16)) % $(17)) * $(16)) + (r15_z % $(16))), dtype=tl.float32) * 2.826)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(272)])
    tl.store($(vmul_1_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(2176))
    r16_value = tl.broadcast_to(0.0, [$(2176)])
    for r16_lane in range($(16)) {
      r16_value = tl.where(tl.load($(vsort_index_ptr) + ((((r16_z / $(128)) % $(17)) * $(128)) + (r16_lane % $(128))), dtype=tl.int64) == tl.cast((r16_z % $(128)), tl.int64), tl.load($(vmul_1_ptr) + ((((r16_z / $(128)) % $(17)) * $(16)) + (r16_lane % $(16))), dtype=tl.float32), r16_value)
    }
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(2176)])
    tl.store($(vscatter_ptr) + r16_z, r16_value)
    z = tl.arange($(2176))
    value = tl.load($(vscatter_ptr) + ((((z / $(128)) % $(17)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(2176)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_router_sigmoid_bias_optimized
