import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_route_expert_bias
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
  let vadd_ptr : RegionName := "vadd"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vgather_1_ptr : RegionName := "vgather_1"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vscatter_ptr : RegionName := "vscatter"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(32768))
    r1_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r1_z / $(128)) % $(256)) * $(128)) + (r1_z % $(128))), dtype=tl.float32))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(32768)])
    tl.store($(vsigmoid_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32768))
    r2_value = (tl.load($(vsigmoid_ptr) + ((((r2_z / $(128)) % $(256)) * $(128)) + (r2_z % $(128))), dtype=tl.float32) + (-0.2 + (r2_z % $(128)) * 0.003149606299212599))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(32768)])
    tl.store($(vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(32768))
    r3_value = (tl.load($(vadd_ptr) + ((((r3_z / $(128)) % $(256)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) != tl.load($(vadd_ptr) + ((((r3_z / $(128)) % $(256)) * $(128)) + (r3_z % $(128))), dtype=tl.float32))
    r3_value = tl.broadcast_to(r3_value, [$(32768)])
    tl.store($(visnan_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = tl.where(tl.load($(visnan_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(vadd_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(vwhere_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_lane = tl.arange($(128))
    r5_row_values = tl.load($(vwhere_ptr) + ((((r5_z[:, None] / $(128)) % $(256)) * $(128)) + (r5_lane[None, :] % $(128))), dtype=tl.float32)
    r5_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r5_other in range($(128)) {
      r5_other_value = tl.load($(vwhere_ptr) + ((((r5_z[:, None] / $(128)) % $(256)) * $(128)) + (r5_other % $(128))), dtype=tl.float32)
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
    r5_value = tl.load($(vwhere_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (tl.cast(r5_value, tl.uint32) % $(128))), dtype=tl.float32)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(vsort_value_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32768))
    r6_lane = tl.arange($(128))
    r6_row_values = tl.load($(vwhere_ptr) + ((((r6_z[:, None] / $(128)) % $(256)) * $(128)) + (r6_lane[None, :] % $(128))), dtype=tl.float32)
    r6_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r6_other in range($(128)) {
      r6_other_value = tl.load($(vwhere_ptr) + ((((r6_z[:, None] / $(128)) % $(256)) * $(128)) + (r6_other % $(128))), dtype=tl.float32)
      r6_other_nan = r6_other_value != r6_other_value
      r6_row_nan = r6_row_values != r6_row_values
      r6_strict = (r6_other_nan & (~r6_row_nan)) | (((~r6_other_nan) & (~r6_row_nan)) & (r6_other_value > r6_row_values))
      r6_equal = (r6_other_nan & r6_row_nan) | (r6_other_value == r6_row_values)
      r6_rank = r6_rank + tl.where(r6_strict | (r6_equal & (r6_other < r6_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r6_selected = r6_rank == tl.cast((r6_z % $(128))[:, None], tl.int64)
    r6_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r6_cutoff in range($(128)) {
      r6_value = r6_value + tl.cast(tl.sum(r6_selected & (r6_lane[None, :] > r6_cutoff), axis=1), tl.int64)
    }
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(vsort_index_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1536))
    r7_value = tl.load($(vsigmoid_ptr) + ((((r7_z / $(6)) % $(256)) * $(128)) + (tl.load($(vsort_index_ptr) + ((((r7_z / $(6)) % $(256)) * $(128)) + (r7_z % $(6))), dtype=tl.int64) % $(128))), dtype=tl.float32)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1536)])
    tl.store($(vgather_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(256))
    r8_lane = tl.arange($(6))
    r8_value = tl.sum(tl.load($(vgather_1_ptr) + ((r8_z[:, None] * $(6)) + r8_lane[None, :]), dtype=tl.float32), axis=1)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(256)])
    tl.store($(vsum_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(256))
    r9_value = (tl.load($(vsum_1_ptr) + r9_z, dtype=tl.float32) + 0.00000000000000000001)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(256)])
    tl.store($(vadd_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(1536))
    r10_value = (tl.load($(vgather_1_ptr) + ((((r10_z / $(6)) % $(256)) * $(6)) + (r10_z % $(6))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r10_z / $(6)) % $(256)), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1536)])
    tl.store($(vdiv_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1536))
    r11_value = (tl.load($(vdiv_ptr) + ((((r11_z / $(6)) % $(256)) * $(6)) + (r11_z % $(6))), dtype=tl.float32) * 2.826)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(1536)])
    tl.store($(vmul_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(32768))
    r12_value = tl.broadcast_to(0.0, [$(32768)])
    for r12_lane in range($(6)) {
      r12_value = tl.where(tl.load($(vsort_index_ptr) + ((((r12_z / $(128)) % $(256)) * $(128)) + (r12_lane % $(128))), dtype=tl.int64) == tl.cast((r12_z % $(128)), tl.int64), tl.load($(vmul_1_ptr) + ((((r12_z / $(128)) % $(256)) * $(6)) + (r12_lane % $(6))), dtype=tl.float32), r12_value)
    }
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(32768)])
    tl.store($(vscatter_ptr) + r12_z, r12_value)
    z = tl.arange($(32768))
    value = tl.load($(vscatter_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let literal_nan_ptr : RegionName := "literal_nan"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vadd_ptr : RegionName := "vadd"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vgather_1_ptr : RegionName := "vgather_1"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vscatter_ptr : RegionName := "vscatter"
  let candidate_literal_nan_ptr : RegionName := "candidate_literal_nan"
  let candidate_vsigmoid_ptr : RegionName := "candidate_vsigmoid"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_visnan_ptr : RegionName := "candidate_visnan"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vsort_value_ptr : RegionName := "candidate_vsort_value"
  let candidate_vsort_index_ptr : RegionName := "candidate_vsort_index"
  let candidate_vgather_1_ptr : RegionName := "candidate_vgather_1"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vscatter_ptr : RegionName := "candidate_vscatter"
  triton {
    r0_z = tl.arange($(1))
    r0_value = tl.bitcast(tl.cast($(2143289344), tl.uint32), tl.float32)
    tl.store($(candidate_literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(32768))
    r1_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r1_z / $(128)) % $(256)) * $(128)) + (r1_z % $(128))), dtype=tl.float32))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(32768)])
    tl.store($(candidate_vsigmoid_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32768))
    r2_value = (tl.load($(candidate_vsigmoid_ptr) + ((((r2_z / $(128)) % $(256)) * $(128)) + (r2_z % $(128))), dtype=tl.float32) + (-0.2 + (r2_z % $(128)) * 0.003149606299212599))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(32768)])
    tl.store($(candidate_vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(32768))
    r3_value = (tl.load($(candidate_vadd_ptr) + ((((r3_z / $(128)) % $(256)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) != tl.load($(candidate_vadd_ptr) + ((((r3_z / $(128)) % $(256)) * $(128)) + (r3_z % $(128))), dtype=tl.float32))
    r3_value = tl.broadcast_to(r3_value, [$(32768)])
    tl.store($(candidate_visnan_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = tl.where(tl.load($(candidate_visnan_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.int1), tl.load($(candidate_literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(candidate_vadd_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(candidate_vwhere_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_lane = tl.arange($(128))
    r5_row_values = tl.load($(candidate_vwhere_ptr) + ((((r5_z[:, None] / $(128)) % $(256)) * $(128)) + (r5_lane[None, :] % $(128))), dtype=tl.float32)
    r5_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r5_other in range($(128)) {
      r5_other_value = tl.load($(candidate_vwhere_ptr) + ((((r5_z[:, None] / $(128)) % $(256)) * $(128)) + (r5_other % $(128))), dtype=tl.float32)
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
    r5_value = tl.load($(candidate_vwhere_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (tl.cast(r5_value, tl.uint32) % $(128))), dtype=tl.float32)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(candidate_vsort_value_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32768))
    r6_lane = tl.arange($(128))
    r6_row_values = tl.load($(candidate_vwhere_ptr) + ((((r6_z[:, None] / $(128)) % $(256)) * $(128)) + (r6_lane[None, :] % $(128))), dtype=tl.float32)
    r6_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r6_other in range($(128)) {
      r6_other_value = tl.load($(candidate_vwhere_ptr) + ((((r6_z[:, None] / $(128)) % $(256)) * $(128)) + (r6_other % $(128))), dtype=tl.float32)
      r6_other_nan = r6_other_value != r6_other_value
      r6_row_nan = r6_row_values != r6_row_values
      r6_strict = (r6_other_nan & (~r6_row_nan)) | (((~r6_other_nan) & (~r6_row_nan)) & (r6_other_value > r6_row_values))
      r6_equal = (r6_other_nan & r6_row_nan) | (r6_other_value == r6_row_values)
      r6_rank = r6_rank + tl.where(r6_strict | (r6_equal & (r6_other < r6_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r6_selected = r6_rank == tl.cast((r6_z % $(128))[:, None], tl.int64)
    r6_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r6_cutoff in range($(128)) {
      r6_value = r6_value + tl.cast(tl.sum(r6_selected & (r6_lane[None, :] > r6_cutoff), axis=1), tl.int64)
    }
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(candidate_vsort_index_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1536))
    r7_value = tl.load($(candidate_vsigmoid_ptr) + ((((r7_z / $(6)) % $(256)) * $(128)) + (tl.load($(candidate_vsort_index_ptr) + ((((r7_z / $(6)) % $(256)) * $(128)) + (r7_z % $(6))), dtype=tl.int64) % $(128))), dtype=tl.float32)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1536)])
    tl.store($(candidate_vgather_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(256))
    r8_lane = tl.arange($(6))
    r8_value = tl.sum(tl.load($(candidate_vgather_1_ptr) + ((r8_z[:, None] * $(6)) + r8_lane[None, :]), dtype=tl.float32), axis=1)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(256)])
    tl.store($(candidate_vsum_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(256))
    r9_value = (tl.load($(candidate_vsum_1_ptr) + r9_z, dtype=tl.float32) + 0.00000000000000000001)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(256)])
    tl.store($(candidate_vadd_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(1536))
    r10_value = (tl.load($(candidate_vgather_1_ptr) + ((((r10_z / $(6)) % $(256)) * $(6)) + (r10_z % $(6))), dtype=tl.float32) / tl.load($(candidate_vadd_1_ptr) + ((r10_z / $(6)) % $(256)), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1536)])
    tl.store($(candidate_vdiv_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1536))
    r11_value = (tl.load($(candidate_vdiv_ptr) + ((((r11_z / $(6)) % $(256)) * $(6)) + (r11_z % $(6))), dtype=tl.float32) * 2.826)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(1536)])
    tl.store($(candidate_vmul_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(32768))
    r12_value = tl.broadcast_to(0.0, [$(32768)])
    for r12_lane in range($(6)) {
      r12_value = tl.where(tl.load($(candidate_vsort_index_ptr) + ((((r12_z / $(128)) % $(256)) * $(128)) + (r12_lane % $(128))), dtype=tl.int64) == tl.cast((r12_z % $(128)), tl.int64), tl.load($(candidate_vmul_1_ptr) + ((((r12_z / $(128)) % $(256)) * $(6)) + (r12_lane % $(6))), dtype=tl.float32), r12_value)
    }
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(32768)])
    tl.store($(candidate_vscatter_ptr) + r12_z, r12_value)
    z = tl.arange($(32768))
    value = tl.load($(candidate_vscatter_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
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
  let vadd_ptr : RegionName := "vadd"
  let visnan_ptr : RegionName := "visnan"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vgather_1_ptr : RegionName := "vgather_1"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vscatter_ptr : RegionName := "vscatter"
  triton {
    r0_z = tl.arange($(1))
    r0_value = 0.0
    tl.store($(literal_nan_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(32768))
    r1_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r1_z / $(128)) % $(256)) * $(128)) + (r1_z % $(128))), dtype=tl.float32))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(32768)])
    tl.store($(vsigmoid_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(32768))
    r2_value = (tl.load($(vsigmoid_ptr) + ((((r2_z / $(128)) % $(256)) * $(128)) + (r2_z % $(128))), dtype=tl.float32) + (-0.2 + (r2_z % $(128)) * 0.003149606299212599))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(32768)])
    tl.store($(vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(32768))
    r3_value = (tl.load($(vadd_ptr) + ((((r3_z / $(128)) % $(256)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) != tl.load($(vadd_ptr) + ((((r3_z / $(128)) % $(256)) * $(128)) + (r3_z % $(128))), dtype=tl.float32))
    r3_value = tl.broadcast_to(r3_value, [$(32768)])
    tl.store($(visnan_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(32768))
    r4_value = tl.where(tl.load($(visnan_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.int1), tl.load($(literal_nan_ptr) + $(0), dtype=tl.float32), tl.load($(vadd_ptr) + ((((r4_z / $(128)) % $(256)) * $(128)) + (r4_z % $(128))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(32768)])
    tl.store($(vwhere_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(32768))
    r5_lane = tl.arange($(128))
    r5_row_values = tl.load($(vwhere_ptr) + ((((r5_z[:, None] / $(128)) % $(256)) * $(128)) + (r5_lane[None, :] % $(128))), dtype=tl.float32)
    r5_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r5_other in range($(128)) {
      r5_other_value = tl.load($(vwhere_ptr) + ((((r5_z[:, None] / $(128)) % $(256)) * $(128)) + (r5_other % $(128))), dtype=tl.float32)
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
    r5_value = tl.load($(vwhere_ptr) + ((((r5_z / $(128)) % $(256)) * $(128)) + (tl.cast(r5_value, tl.uint32) % $(128))), dtype=tl.float32)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(32768)])
    tl.store($(vsort_value_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(32768))
    r6_lane = tl.arange($(128))
    r6_row_values = tl.load($(vwhere_ptr) + ((((r6_z[:, None] / $(128)) % $(256)) * $(128)) + (r6_lane[None, :] % $(128))), dtype=tl.float32)
    r6_rank = tl.full([$(32768), $(128)], $(0), dtype=tl.int64)
    for r6_other in range($(128)) {
      r6_other_value = tl.load($(vwhere_ptr) + ((((r6_z[:, None] / $(128)) % $(256)) * $(128)) + (r6_other % $(128))), dtype=tl.float32)
      r6_other_nan = r6_other_value != r6_other_value
      r6_row_nan = r6_row_values != r6_row_values
      r6_strict = (r6_other_nan & (~r6_row_nan)) | (((~r6_other_nan) & (~r6_row_nan)) & (r6_other_value > r6_row_values))
      r6_equal = (r6_other_nan & r6_row_nan) | (r6_other_value == r6_row_values)
      r6_rank = r6_rank + tl.where(r6_strict | (r6_equal & (r6_other < r6_lane[None, :])), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r6_selected = r6_rank == tl.cast((r6_z % $(128))[:, None], tl.int64)
    r6_value = tl.full([$(32768)], $(0), dtype=tl.int64)
    for r6_cutoff in range($(128)) {
      r6_value = r6_value + tl.cast(tl.sum(r6_selected & (r6_lane[None, :] > r6_cutoff), axis=1), tl.int64)
    }
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(vsort_index_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1536))
    r7_value = tl.load($(vsigmoid_ptr) + ((((r7_z / $(6)) % $(256)) * $(128)) + (tl.load($(vsort_index_ptr) + ((((r7_z / $(6)) % $(256)) * $(128)) + (r7_z % $(6))), dtype=tl.int64) % $(128))), dtype=tl.float32)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1536)])
    tl.store($(vgather_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(256))
    r8_lane = tl.arange($(6))
    r8_value = tl.sum(tl.load($(vgather_1_ptr) + ((r8_z[:, None] * $(6)) + r8_lane[None, :]), dtype=tl.float32), axis=1)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(256)])
    tl.store($(vsum_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(256))
    r9_value = (tl.load($(vsum_1_ptr) + r9_z, dtype=tl.float32) + 0.00000000000000000001)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(256)])
    tl.store($(vadd_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(1536))
    r10_value = (tl.load($(vgather_1_ptr) + ((((r10_z / $(6)) % $(256)) * $(6)) + (r10_z % $(6))), dtype=tl.float32) / tl.load($(vadd_1_ptr) + ((r10_z / $(6)) % $(256)), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1536)])
    tl.store($(vdiv_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1536))
    r11_value = (tl.load($(vdiv_ptr) + ((((r11_z / $(6)) % $(256)) * $(6)) + (r11_z % $(6))), dtype=tl.float32) * 2.826)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(1536)])
    tl.store($(vmul_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(32768))
    r12_value = tl.broadcast_to(0.0, [$(32768)])
    for r12_lane in range($(6)) {
      r12_value = tl.where(tl.load($(vsort_index_ptr) + ((((r12_z / $(128)) % $(256)) * $(128)) + (r12_lane % $(128))), dtype=tl.int64) == tl.cast((r12_z % $(128)), tl.int64), tl.load($(vmul_1_ptr) + ((((r12_z / $(128)) % $(256)) * $(6)) + (r12_lane % $(6))), dtype=tl.float32), r12_value)
    }
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(32768)])
    tl.store($(vscatter_ptr) + r12_z, r12_value)
    z = tl.arange($(32768))
    value = tl.load($(vscatter_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_route_expert_bias
