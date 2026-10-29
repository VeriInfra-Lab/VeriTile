import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_linear_fwd
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
  let vmm_ptr : RegionName := "vmm"
  triton {
    r0_z = tl.arange($(4194304))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(2048)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(4194304)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(256))
    r1_col = tl.arange($(2048))
    r1_lane = tl.arange($(2048))
    r1_left = tl.load($(input0_ptr) + ((((((r1_row[:, None] * $(2048)) + r1_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r1_row[:, None] * $(2048)) + r1_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r1_row[:, None] * $(2048)) + r1_lane[None, :]) % $(2048))), dtype=tl.bfloat16)
    r1_right = tl.load($(vmul_ptr) + (((r1_col[None, :] % $(2048)) * $(2048)) + (r1_lane[:, None] % $(2048))), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r1_row[:, None] * $(2048)) + r1_col[None, :]), r1_value)
    z = tl.arange($(524288))
    value = tl.load($(vmm_ptr) + ((((z / $(262144)) % $(2)) * $(262144)) + (((z / $(2048)) % $(128)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmm_ptr : RegionName := "vmm"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  triton {
    r0_z = tl.arange($(4194304))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(2048)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(4194304)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(256))
    r1_col = tl.arange($(2048))
    r1_lane = tl.arange($(2048))
    r1_left = tl.load($(input0_ptr) + ((((((r1_row[:, None] * $(2048)) + r1_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r1_row[:, None] * $(2048)) + r1_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r1_row[:, None] * $(2048)) + r1_lane[None, :]) % $(2048))), dtype=tl.bfloat16)
    r1_right = tl.load($(candidate_vmul_ptr) + (((r1_col[None, :] % $(2048)) * $(2048)) + (r1_lane[:, None] % $(2048))), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r1_row[:, None] * $(2048)) + r1_col[None, :]), r1_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vmm_ptr) + ((((z / $(262144)) % $(2)) * $(262144)) + (((z / $(2048)) % $(128)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_linear_fwd
