import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_linear_wgrad
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
  let vmm_1_ptr : RegionName := "vmm_1"
  triton {
    r0_row = tl.arange($(2048))
    r0_col = tl.arange($(2048))
    r0_lane = tl.arange($(256))
    r0_left = tl.load($(input1_ptr) + ((((((r0_lane[None, :] * $(2048)) + r0_row[:, None]) / $(262144)) % $(2)) * $(262144)) + (((((r0_lane[None, :] * $(2048)) + r0_row[:, None]) / $(2048)) % $(128)) * $(2048)) + (((r0_lane[None, :] * $(2048)) + r0_row[:, None]) % $(2048))), dtype=tl.bfloat16)
    r0_right = tl.load($(input0_ptr) + ((((((r0_lane[:, None] * $(2048)) + r0_col[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r0_lane[:, None] * $(2048)) + r0_col[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r0_lane[:, None] * $(2048)) + r0_col[None, :]) % $(2048))), dtype=tl.bfloat16)
    r0_value = tl.dot(r0_left, r0_right)
    r0_value = r0_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r0_row[:, None] * $(2048)) + r0_col[None, :]), r0_value)
    z = tl.arange($(4194304))
    value = tl.load($(vmm_1_ptr) + ((((z / $(2048)) % $(2048)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(4194304)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  triton {
    r0_row = tl.arange($(2048))
    r0_col = tl.arange($(2048))
    r0_lane = tl.arange($(256))
    r0_left = tl.load($(input1_ptr) + ((((((r0_lane[None, :] * $(2048)) + r0_row[:, None]) / $(262144)) % $(2)) * $(262144)) + (((((r0_lane[None, :] * $(2048)) + r0_row[:, None]) / $(2048)) % $(128)) * $(2048)) + (((r0_lane[None, :] * $(2048)) + r0_row[:, None]) % $(2048))), dtype=tl.bfloat16)
    r0_right = tl.load($(input0_ptr) + ((((((r0_lane[:, None] * $(2048)) + r0_col[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r0_lane[:, None] * $(2048)) + r0_col[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r0_lane[:, None] * $(2048)) + r0_col[None, :]) % $(2048))), dtype=tl.bfloat16)
    r0_value = tl.dot(r0_left, r0_right)
    r0_value = r0_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r0_row[:, None] * $(2048)) + r0_col[None, :]), r0_value)
    z = tl.arange($(4194304))
    value = tl.load($(candidate_vmm_2_ptr) + ((((z / $(2048)) % $(2048)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(4194304)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_linear_wgrad
