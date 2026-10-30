import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_indices_to_multihot_probs
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vne_ptr : RegionName := "vne"
  let vlt_ptr : RegionName := "vlt"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vwhere_ptr : RegionName := "vwhere"
  let vscatter_1_ptr : RegionName := "vscatter_1"
  triton {
    r0_z = tl.arange($(512))
    r0_value = (tl.load($(input0_ptr) + ((((r0_z / $(4)) % $(128)) * $(4)) + (r0_z % $(4))), dtype=tl.int64) != (-tl.full([], $(1), dtype=tl.int64)))
    r0_value = tl.broadcast_to(r0_value, [$(512)])
    tl.store($(vne_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(512))
    r1_value = (tl.load($(input0_ptr) + ((((r1_z / $(4)) % $(128)) * $(4)) + (r1_z % $(4))), dtype=tl.int64) < tl.full([], $(16), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(512)])
    tl.store($(vlt_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(512))
    r2_value = (tl.load($(vne_ptr) + ((((r2_z / $(4)) % $(128)) * $(4)) + (r2_z % $(4))), dtype=tl.int1) & tl.load($(vlt_ptr) + ((((r2_z / $(4)) % $(128)) * $(4)) + (r2_z % $(4))), dtype=tl.int1))
    r2_value = tl.broadcast_to(r2_value, [$(512)])
    tl.store($(vbitwise_and_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(512))
    r3_value = tl.where(tl.load($(vbitwise_and_ptr) + ((((r3_z / $(4)) % $(128)) * $(4)) + (r3_z % $(4))), dtype=tl.int1), tl.load($(input0_ptr) + ((((r3_z / $(4)) % $(128)) * $(4)) + (r3_z % $(4))), dtype=tl.int64), tl.full([], $(16), dtype=tl.int64))
    r3_value = tl.broadcast_to(r3_value, [$(512)])
    tl.store($(vwhere_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(2176))
    r4_value = tl.broadcast_to(tl.cast(0.0, tl.bfloat16), [$(2176)])
    for r4_lane in range($(4)) {
      r4_value = tl.where(tl.load($(vwhere_ptr) + ((((r4_z / $(17)) % $(128)) * $(4)) + (r4_lane % $(4))), dtype=tl.int64) == tl.cast((r4_z % $(17)), tl.int64), tl.load($(input1_ptr) + ((((r4_z / $(17)) % $(128)) * $(4)) + (r4_lane % $(4))), dtype=tl.bfloat16), r4_value)
    }
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(2176)])
    tl.store($(vscatter_1_ptr) + r4_z, r4_value)
    z = tl.arange($(2048))
    value = tl.load($(vscatter_1_ptr) + ((((z / $(16)) % $(128)) * $(17)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2048)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vwhere_ptr : RegionName := "vwhere"
  let candidate_veq_ptr : RegionName := "candidate_veq"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  triton {
    r0_z = tl.arange($(8192))
    r0_value = (tl.load($(input0_ptr) + ((((r0_z / $(64)) % $(128)) * $(4)) + ((r0_z / $(16)) % $(4))), dtype=tl.int64) == tl.cast((r0_z % $(16)), tl.int64))
    r0_value = tl.broadcast_to(r0_value, [$(8192)])
    tl.store($(candidate_veq_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(8192))
    r1_value = tl.where(tl.load($(candidate_veq_ptr) + ((((r1_z / $(64)) % $(128)) * $(64)) + (((r1_z / $(16)) % $(4)) * $(16)) + (r1_z % $(16))), dtype=tl.int1), tl.load($(input1_ptr) + ((((r1_z / $(64)) % $(128)) * $(4)) + ((r1_z / $(16)) % $(4))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(8192)])
    tl.store($(candidate_vwhere_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(2048))
    r2_lane = tl.arange($(4))
    r2_value = tl.sum(tl.load($(candidate_vwhere_ptr) + ((((r2_z[:, None] / $(16)) % $(128)) * $(64)) + (r2_lane[None, :] * $(16)) + (r2_z[:, None] % $(16))), dtype=tl.bfloat16), axis=1)
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(2048)])
    tl.store($(candidate_vsum_1_ptr) + r2_z, r2_value)
    z = tl.arange($(2048))
    value = tl.load($(candidate_vsum_1_ptr) + ((((z / $(16)) % $(128)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2048)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_indices_to_multihot_probs
