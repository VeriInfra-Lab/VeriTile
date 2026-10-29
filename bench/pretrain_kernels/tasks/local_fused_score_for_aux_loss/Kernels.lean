import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vadd_ptr : RegionName := "vadd"
  let vdiv_ptr : RegionName := "vdiv"
  triton {
    r0_z = tl.arange($(32768))
    r0_value = tl.sigmoid(tl.load($(input0_ptr) + ((((r0_z / $(128)) % $(256)) * $(128)) + (r0_z % $(128))), dtype=tl.float32))
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(32768)])
    tl.store($(vsigmoid_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_lane = tl.arange($(128))
    r1_value = tl.sum(tl.load($(vsigmoid_ptr) + ((r1_z[:, None] * $(128)) + r1_lane[None, :]), dtype=tl.float32), axis=1)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(vsum_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_value = (tl.load($(vsum_1_ptr) + r2_z, dtype=tl.float32) + 0.00000000000000000001)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(32768))
    r3_value = (tl.load($(vsigmoid_ptr) + ((((r3_z / $(128)) % $(256)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) / tl.load($(vadd_ptr) + ((r3_z / $(128)) % $(256)), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(32768)])
    tl.store($(vdiv_ptr) + r3_z, r3_value)
    z = tl.arange($(32768))
    value = tl.load($(vdiv_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
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
  let candidate_vsigmoid_ptr : RegionName := "candidate_vsigmoid"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
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
    z = tl.arange($(32768))
    value = tl.load($(candidate_vdiv_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss
