import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mtp_row_boundaries_closed_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let veq_ptr : RegionName := "veq"
  let vwhere_ptr : RegionName := "vwhere"
  let vsort_value_ptr : RegionName := "vsort_value"
  let vsort_index_ptr : RegionName := "vsort_index"
  let vcat_ptr : RegionName := "vcat"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  triton {
    r0_z = tl.arange($(127))
    r0_value = (tl.load($(input0_ptr) + r0_z, dtype=tl.int64) == tl.full([], $(2), dtype=tl.int64))
    r0_value = tl.broadcast_to(r0_value, [$(127)])
    tl.store($(veq_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(127))
    r1_value = tl.where(tl.load($(veq_ptr) + r1_z, dtype=tl.int1), (tl.full([], $(1), dtype=tl.int64) + tl.cast(r1_z, tl.int64)), tl.full([], $(129), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(127)])
    tl.store($(vwhere_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(127))
    r2_lane = tl.arange($(127))
    r2_row_values = tl.load($(vwhere_ptr) + (r2_lane[None, :] % $(127)), dtype=tl.int64)
    r2_rank = tl.full([$(127), $(127)], $(0), dtype=tl.int64)
    for r2_other in range($(127)) {
      r2_rank = r2_rank + tl.where(((tl.load($(vwhere_ptr) + (r2_other % $(127)), dtype=tl.int64) < r2_row_values) | ((tl.load($(vwhere_ptr) + (r2_other % $(127)), dtype=tl.int64) == r2_row_values) & (r2_other < r2_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r2_selected = r2_rank == tl.cast(r2_z[:, None], tl.int64)
    r2_value = tl.full([$(127)], $(0), dtype=tl.int64)
    for r2_cutoff in range($(127)) {
      r2_value = r2_value + tl.cast(tl.sum(r2_selected & (r2_lane[None, :] > r2_cutoff), axis=1), tl.int64)
    }
    r2_value = tl.load($(vwhere_ptr) + tl.cast(r2_value, tl.uint32), dtype=tl.int64)
    r2_value = tl.broadcast_to(r2_value, [$(127)])
    tl.store($(vsort_value_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(127))
    r3_lane = tl.arange($(127))
    r3_row_values = tl.load($(vwhere_ptr) + (r3_lane[None, :] % $(127)), dtype=tl.int64)
    r3_rank = tl.full([$(127), $(127)], $(0), dtype=tl.int64)
    for r3_other in range($(127)) {
      r3_rank = r3_rank + tl.where(((tl.load($(vwhere_ptr) + (r3_other % $(127)), dtype=tl.int64) < r3_row_values) | ((tl.load($(vwhere_ptr) + (r3_other % $(127)), dtype=tl.int64) == r3_row_values) & (r3_other < r3_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r3_selected = r3_rank == tl.cast(r3_z[:, None], tl.int64)
    r3_value = tl.full([$(127)], $(0), dtype=tl.int64)
    for r3_cutoff in range($(127)) {
      r3_value = r3_value + tl.cast(tl.sum(r3_selected & (r3_lane[None, :] > r3_cutoff), axis=1), tl.int64)
    }
    r3_value = tl.broadcast_to(r3_value, [$(127)])
    tl.store($(vsort_index_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(4))
    r4_value = tl.where(r4_z < $(1), tl.full([], $(0), dtype=tl.int64), tl.where(r4_z < $(3), tl.load($(vsort_value_ptr) + ((r4_z - $(1)) % $(127)), dtype=tl.int64), tl.full([], $(128), dtype=tl.int64)))
    r4_value = tl.broadcast_to(r4_value, [$(4)])
    tl.store($(vcat_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(4))
    r5_value = tl.toReal(tl.load($(vcat_ptr) + r5_z, dtype=tl.int64))
    r5_value = r5_value.to(tl.float64)
    r5_value = tl.broadcast_to(r5_value, [$(4)])
    tl.store($(vto_copy_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(4))
    r6_value = tl.load($(vto_copy_ptr) + r6_z, dtype=tl.float64)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(4)])
    tl.store($(vto_copy_1_ptr) + r6_z, r6_value)
    z = tl.arange($(4))
    value = tl.load($(vto_copy_1_ptr) + z, dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(4)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let veq_ptr : RegionName := "veq"
  let vwhere_ptr : RegionName := "vwhere"
  let vcat_ptr : RegionName := "vcat"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let candidate_veq_ptr : RegionName := "candidate_veq"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vcumsum_ptr : RegionName := "candidate_vcumsum"
  let candidate_veq_1_ptr : RegionName := "candidate_veq_1"
  let candidate_vbitwise_and_ptr : RegionName := "candidate_vbitwise_and"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_veq_2_ptr : RegionName := "candidate_veq_2"
  let candidate_vbitwise_and_1_ptr : RegionName := "candidate_vbitwise_and_1"
  let candidate_vwhere_1_ptr : RegionName := "candidate_vwhere_1"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  triton {
    r0_z = tl.arange($(127))
    r0_value = (tl.load($(input0_ptr) + r0_z, dtype=tl.int64) == tl.full([], $(2), dtype=tl.int64))
    r0_value = tl.broadcast_to(r0_value, [$(127)])
    tl.store($(candidate_veq_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(127))
    r1_value = tl.where(tl.load($(candidate_veq_ptr) + r1_z, dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(127)])
    tl.store($(candidate_vto_copy_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(127))
    r2_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(127)])
    for r2_lane in range($(127)) {
      r2_value = tl.where(r2_lane <= r2_z, (r2_value + tl.load($(candidate_vto_copy_ptr) + (r2_lane % $(127)), dtype=tl.int64)), r2_value)
    }
    r2_value = tl.broadcast_to(r2_value, [$(127)])
    tl.store($(candidate_vcumsum_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(127))
    r3_value = (tl.load($(candidate_vcumsum_ptr) + r3_z, dtype=tl.int64) == tl.full([], $(1), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(127)])
    tl.store($(candidate_veq_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(127))
    r4_value = (tl.load($(candidate_veq_ptr) + r4_z, dtype=tl.int1) & tl.load($(candidate_veq_1_ptr) + r4_z, dtype=tl.int1))
    r4_value = tl.broadcast_to(r4_value, [$(127)])
    tl.store($(candidate_vbitwise_and_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(127))
    r5_value = tl.where(tl.load($(candidate_vbitwise_and_ptr) + r5_z, dtype=tl.int1), (tl.full([], $(1), dtype=tl.int64) + tl.cast(r5_z, tl.int64)), tl.full([], $(0), dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(127)])
    tl.store($(candidate_vwhere_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(1))
    r6_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r6_lane in range($(127)) {
      r6_value = r6_value + tl.load($(candidate_vwhere_ptr) + r6_lane, dtype=tl.int64)
    }
    r6_value = tl.broadcast_to(r6_value, [$(1)])
    tl.store($(candidate_vsum_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(127))
    r7_value = (tl.load($(candidate_vcumsum_ptr) + r7_z, dtype=tl.int64) == tl.full([], $(2), dtype=tl.int64))
    r7_value = tl.broadcast_to(r7_value, [$(127)])
    tl.store($(candidate_veq_2_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(127))
    r8_value = (tl.load($(candidate_veq_ptr) + r8_z, dtype=tl.int1) & tl.load($(candidate_veq_2_ptr) + r8_z, dtype=tl.int1))
    r8_value = tl.broadcast_to(r8_value, [$(127)])
    tl.store($(candidate_vbitwise_and_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(127))
    r9_value = tl.where(tl.load($(candidate_vbitwise_and_1_ptr) + r9_z, dtype=tl.int1), (tl.full([], $(1), dtype=tl.int64) + tl.cast(r9_z, tl.int64)), tl.full([], $(0), dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(127)])
    tl.store($(candidate_vwhere_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(1))
    r10_value = tl.full([$(1)], $(0), dtype=tl.int64)
    for r10_lane in range($(127)) {
      r10_value = r10_value + tl.load($(candidate_vwhere_1_ptr) + r10_lane, dtype=tl.int64)
    }
    r10_value = tl.broadcast_to(r10_value, [$(1)])
    tl.store($(candidate_vsum_2_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(2))
    r11_value = tl.where(r11_z < $(1), tl.load($(candidate_vsum_1_ptr) + $(0), dtype=tl.int64), tl.load($(candidate_vsum_2_ptr) + $(0), dtype=tl.int64))
    r11_value = tl.broadcast_to(r11_value, [$(2)])
    tl.store($(candidate_vstack_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(4))
    r12_value = tl.where(r12_z < $(1), tl.full([], $(0), dtype=tl.int64), tl.where(r12_z < $(3), tl.load($(candidate_vstack_ptr) + ((r12_z - $(1)) % $(2)), dtype=tl.int64), tl.full([], $(128), dtype=tl.int64)))
    r12_value = tl.broadcast_to(r12_value, [$(4)])
    tl.store($(candidate_vcat_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(4))
    r13_value = tl.toReal(tl.load($(candidate_vcat_ptr) + r13_z, dtype=tl.int64))
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(4)])
    tl.store($(candidate_vto_copy_1_ptr) + r13_z, r13_value)
    z = tl.arange($(4))
    value = tl.load($(candidate_vto_copy_1_ptr) + z, dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(4)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mtp_row_boundaries_closed_fwd
