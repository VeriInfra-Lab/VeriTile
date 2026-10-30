import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_probs_scale_dx
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
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  triton {
    r0_z = tl.arange($(8388608))
    r0_value = tl.load($(input1_ptr) + ((((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(8388608)])
    tl.store($(vto_copy_2_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(8388608))
    r1_value = (tl.load($(vto_copy_2_ptr) + ((((r1_z / $(2048)) % $(4096)) * $(2048)) + (r1_z % $(2048))), dtype=tl.float32) * tl.load($(input0_ptr) + ((r1_z / $(2048)) % $(4096)), dtype=tl.float32))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(8388608)])
    tl.store($(vmul_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(8388608))
    r2_value = tl.load($(vmul_1_ptr) + ((((r2_z / $(2048)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32)
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(8388608)])
    tl.store($(vto_copy_3_ptr) + r2_z, r2_value)
    z = tl.arange($(8388608))
    value = tl.load($(vto_copy_3_ptr) + ((((z / $(2048)) % $(4096)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(8388608)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  triton {
    r0_z = tl.arange($(8388608))
    r0_value = tl.load($(input1_ptr) + ((((r0_z / $(2048)) % $(4096)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(8388608)])
    tl.store($(candidate_vto_copy_2_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(8388608))
    r1_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r1_z / $(2048)) % $(4096)) * $(2048)) + (r1_z % $(2048))), dtype=tl.float32) * tl.load($(input0_ptr) + ((r1_z / $(2048)) % $(4096)), dtype=tl.float32))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(8388608)])
    tl.store($(candidate_vmul_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(8388608))
    r2_value = tl.load($(candidate_vmul_1_ptr) + ((((r2_z / $(2048)) % $(4096)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32)
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(8388608)])
    tl.store($(candidate_vto_copy_3_ptr) + r2_z, r2_value)
    z = tl.arange($(8388608))
    value = tl.load($(candidate_vto_copy_3_ptr) + ((((z / $(2048)) % $(4096)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(8388608)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_probs_scale_dx
