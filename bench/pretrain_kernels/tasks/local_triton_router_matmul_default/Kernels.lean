import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_triton_router_matmul_default
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
    r0_z = tl.arange($(16705))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(257)) % $(65)) * $(257)) + (r0_z % $(257))), dtype=tl.float32) * 0.02)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(16705)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(65))
    r1_col = tl.arange($(65))
    r1_lane = tl.arange($(257))
    r1_left = tl.load($(input0_ptr) + (((r1_row[:, None] % $(65)) * $(257)) + (r1_lane[None, :] % $(257))), dtype=tl.float32)
    r1_right = tl.load($(vmul_ptr) + (((r1_col[None, :] % $(65)) * $(257)) + (r1_lane[:, None] % $(257))), dtype=tl.float32)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.float32)
    tl.store($(vmm_ptr) + ((r1_row[:, None] * $(65)) + r1_col[None, :]), r1_value)
    z = tl.arange($(4225))
    value = tl.load($(vmm_ptr) + ((((z / $(65)) % $(65)) * $(65)) + (z % $(65))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(4225)])
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
    r0_z = tl.arange($(16705))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(257)) % $(65)) * $(257)) + (r0_z % $(257))), dtype=tl.float32) * 0.02)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(16705)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(65))
    r1_col = tl.arange($(65))
    r1_lane = tl.arange($(257))
    r1_left = tl.load($(input0_ptr) + (((r1_row[:, None] % $(65)) * $(257)) + (r1_lane[None, :] % $(257))), dtype=tl.float32)
    r1_right = tl.load($(candidate_vmul_ptr) + (((r1_col[None, :] % $(65)) * $(257)) + (r1_lane[:, None] % $(257))), dtype=tl.float32)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.float32)
    tl.store($(candidate_vmm_ptr) + ((r1_row[:, None] * $(65)) + r1_col[None, :]), r1_value)
    z = tl.arange($(4225))
    value = tl.load($(candidate_vmm_ptr) + ((((z / $(65)) % $(65)) * $(65)) + (z % $(65))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(4225)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_triton_router_matmul_default
