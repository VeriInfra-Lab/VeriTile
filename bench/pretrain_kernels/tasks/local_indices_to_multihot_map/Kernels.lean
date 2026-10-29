import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_indices_to_multihot_map
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vne_ptr : RegionName := "vne"
  let vlt_ptr : RegionName := "vlt"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vwhere_ptr : RegionName := "vwhere"
  let vscatter_ptr : RegionName := "vscatter"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
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
    r4_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(2176)])
    for r4_lane in range($(4)) {
      r4_value = tl.where(tl.load($(vwhere_ptr) + ((((r4_z / $(17)) % $(128)) * $(4)) + (r4_lane % $(4))), dtype=tl.int64) == tl.cast((r4_z % $(17)), tl.int64), tl.full([], $(1), dtype=tl.int64), r4_value)
    }
    r4_value = tl.broadcast_to(r4_value, [$(2176)])
    tl.store($(vscatter_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(2048))
    r5_value = (tl.load($(vscatter_ptr) + ((((r5_z / $(16)) % $(128)) * $(17)) + (r5_z % $(16))), dtype=tl.int64) != tl.full([], $(0), dtype=tl.int64))
    r5_value = tl.broadcast_to(r5_value, [$(2048)])
    tl.store($(vto_copy_1_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2048))
    r6_value = tl.where(tl.load($(vto_copy_1_ptr) + ((((r6_z / $(16)) % $(128)) * $(16)) + (r6_z % $(16))), dtype=tl.int1), tl.cast(1.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(2048)])
    tl.store($(vto_copy_2_ptr) + r6_z, r6_value)
    z = tl.arange($(2048))
    value = tl.load($(vto_copy_2_ptr) + ((((z / $(16)) % $(128)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2048)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let candidate_veq_ptr : RegionName := "candidate_veq"
  let candidate_vany_1_ptr : RegionName := "candidate_vany_1"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  triton {
    r0_z = tl.arange($(8192))
    r0_value = (tl.load($(input0_ptr) + ((((r0_z / $(64)) % $(128)) * $(4)) + ((r0_z / $(16)) % $(4))), dtype=tl.int64) == tl.cast((r0_z % $(16)), tl.int64))
    r0_value = tl.broadcast_to(r0_value, [$(8192)])
    tl.store($(candidate_veq_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(2048))
    r1_lane = tl.arange($(4))
    r1_value = tl.sum(tl.load($(candidate_veq_ptr) + ((((r1_z[:, None] / $(16)) % $(128)) * $(64)) + ((r1_lane[None, :] % $(4)) * $(16)) + (r1_z[:, None] % $(16))), dtype=tl.int1), axis=1) > $(0)
    r1_value = tl.broadcast_to(r1_value, [$(2048)])
    tl.store($(candidate_vany_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(2048))
    r2_value = tl.where(tl.load($(candidate_vany_1_ptr) + ((((r2_z / $(16)) % $(128)) * $(16)) + (r2_z % $(16))), dtype=tl.int1), tl.cast(1.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(2048)])
    tl.store($(candidate_vto_copy_ptr) + r2_z, r2_value)
    z = tl.arange($(2048))
    value = tl.load($(candidate_vto_copy_ptr) + ((((z / $(16)) % $(128)) * $(16)) + (z % $(16))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(2048)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_indices_to_multihot_map
