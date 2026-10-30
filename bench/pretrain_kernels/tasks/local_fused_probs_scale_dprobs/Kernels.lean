import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_probs_scale_dprobs
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
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vsum_1_ptr : RegionName := "vsum_1"
  triton {
    r0_z = tl.arange($(8388608))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(8388608)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(8388608))
    r1_value = tl.load($(input1_ptr) + ((((r1_z / $(2048)) % $(4096)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(8388608)])
    tl.store($(vto_copy_2_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(8388608))
    r2_value = (tl.load($(vto_copy_2_ptr) + ((((r2_z / $(2048)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32) * tl.load($(vto_copy_ptr) + ((((r2_z / $(2048)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(8388608)])
    tl.store($(vmul_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(4096))
    r3_lane = tl.arange($(2048))
    r3_value = tl.sum(tl.load($(vmul_1_ptr) + ((r3_z[:, None] * $(2048)) + r3_lane[None, :]), dtype=tl.float32), axis=1)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(4096)])
    tl.store($(vsum_1_ptr) + r3_z, r3_value)
    z = tl.arange($(4096))
    value = tl.load($(vsum_1_ptr) + z, dtype=tl.float32)
    value = tl.broadcast_to(value, [$(4096)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vsum_1_ptr : RegionName := "vsum_1"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  triton {
    r0_z = tl.arange($(8388608))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(8388608)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(8388608))
    r1_value = tl.load($(input1_ptr) + ((((r1_z / $(2048)) % $(4096)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(8388608)])
    tl.store($(candidate_vto_copy_2_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(8388608))
    r2_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r2_z / $(2048)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + ((((r2_z / $(2048)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(8388608)])
    tl.store($(candidate_vmul_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(4096))
    r3_lane = tl.arange($(2048))
    r3_value = tl.sum(tl.load($(candidate_vmul_1_ptr) + ((r3_z[:, None] * $(2048)) + r3_lane[None, :]), dtype=tl.float32), axis=1)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(4096)])
    tl.store($(candidate_vsum_1_ptr) + r3_z, r3_value)
    z = tl.arange($(4096))
    value = tl.load($(candidate_vsum_1_ptr) + z, dtype=tl.float32)
    value = tl.broadcast_to(value, [$(4096)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_probs_scale_dprobs
