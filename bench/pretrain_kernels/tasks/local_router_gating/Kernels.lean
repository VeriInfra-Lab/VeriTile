import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_router_gating
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vmm_ptr : RegionName := "vmm"
  triton {
    r0_z = tl.arange($(262144))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(128)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(262144)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(524288))
    r1_value = tl.load($(input0_ptr) + ((((((((r1_z / $(2048)) % $(256)) * $(2048)) + (r1_z % $(2048))) / $(262144)) % $(2)) * $(262144)) + ((((r1_z / $(2048)) % $(256)) % $(128)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(524288)])
    tl.store($(vto_copy_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(262144))
    r2_value = tl.load($(vmul_ptr) + ((((r2_z / $(2048)) % $(128)) * $(2048)) + (r2_z % $(2048))), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(262144)])
    tl.store($(vto_copy_1_ptr) + r2_z, r2_value)
    r3_row = tl.arange($(256))
    r3_col = tl.arange($(128))
    r3_lane = tl.arange($(2048))
    r3_left = tl.load($(vto_copy_ptr) + (((r3_row[:, None] % $(256)) * $(2048)) + (r3_lane[None, :] % $(2048))), dtype=tl.float32)
    r3_right = tl.load($(vto_copy_1_ptr) + (((r3_col[None, :] % $(128)) * $(2048)) + (r3_lane[:, None] % $(2048))), dtype=tl.float32)
    r3_value = tl.dot(r3_left, r3_right)
    r3_value = r3_value.to(tl.float32)
    tl.store($(vmm_ptr) + ((r3_row[:, None] * $(128)) + r3_col[None, :]), r3_value)
    z = tl.arange($(32768))
    value = tl.load($(vmm_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vmm_ptr : RegionName := "vmm"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  triton {
    r0_z = tl.arange($(262144))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(128)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(262144)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(524288))
    r1_value = tl.load($(input0_ptr) + ((((((((r1_z / $(2048)) % $(256)) * $(2048)) + (r1_z % $(2048))) / $(262144)) % $(2)) * $(262144)) + ((((r1_z / $(2048)) % $(256)) % $(128)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(524288)])
    tl.store($(candidate_vto_copy_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(262144))
    r2_value = tl.load($(candidate_vmul_ptr) + ((((r2_z / $(2048)) % $(128)) * $(2048)) + (r2_z % $(2048))), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(262144)])
    tl.store($(candidate_vto_copy_1_ptr) + r2_z, r2_value)
    r3_row = tl.arange($(256))
    r3_col = tl.arange($(128))
    r3_lane = tl.arange($(2048))
    r3_left = tl.load($(candidate_vto_copy_ptr) + (((r3_row[:, None] % $(256)) * $(2048)) + (r3_lane[None, :] % $(2048))), dtype=tl.float32)
    r3_right = tl.load($(candidate_vto_copy_1_ptr) + (((r3_col[None, :] % $(128)) * $(2048)) + (r3_lane[:, None] % $(2048))), dtype=tl.float32)
    r3_value = tl.dot(r3_left, r3_right)
    r3_value = r3_value.to(tl.float32)
    tl.store($(candidate_vmm_ptr) + ((r3_row[:, None] * $(128)) + r3_col[None, :]), r3_value)
    z = tl.arange($(32768))
    value = tl.load($(candidate_vmm_ptr) + ((((z / $(128)) % $(256)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(32768)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_router_gating
