import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_triton_router_matmul_production
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vaddmm_ptr : RegionName := "vaddmm"
  triton {
    r0_z = tl.arange($(786432))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(384)) * $(2048)) + (r0_z % $(2048))), dtype=tl.float32) * 0.02)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(786432)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(384))
    r1_value = (tl.load($(parameter1_ptr) + r1_z, dtype=tl.float32) * 0.02)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(384)])
    tl.store($(vmul_1_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(128))
    r2_col = tl.arange($(384))
    r2_lane = tl.arange($(2048))
    r2_left = tl.load($(input0_ptr) + (((r2_row[:, None] % $(128)) * $(2048)) + (r2_lane[None, :] % $(2048))), dtype=tl.float32)
    r2_right = tl.load($(vmul_ptr) + (((r2_col[None, :] % $(384)) * $(2048)) + (r2_lane[:, None] % $(2048))), dtype=tl.float32)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value + tl.load($(vmul_1_ptr) + (r2_col[None, :] % $(384)), dtype=tl.float32) * $(1)
    r2_value = r2_value.to(tl.float32)
    tl.store($(vaddmm_ptr) + ((r2_row[:, None] * $(384)) + r2_col[None, :]), r2_value)
    z = tl.arange($(49152))
    value = tl.load($(vaddmm_ptr) + ((((z / $(384)) % $(128)) * $(384)) + (z % $(384))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(49152)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  triton {
    r0_z = tl.arange($(786432))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(384)) * $(2048)) + (r0_z % $(2048))), dtype=tl.float32) * 0.02)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(786432)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(384))
    r1_value = (tl.load($(parameter1_ptr) + r1_z, dtype=tl.float32) * 0.02)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(384)])
    tl.store($(candidate_vmul_1_ptr) + r1_z, r1_value)
    r2_row = tl.arange($(128))
    r2_col = tl.arange($(384))
    r2_lane = tl.arange($(2048))
    r2_left = tl.load($(input0_ptr) + (((r2_row[:, None] % $(128)) * $(2048)) + (r2_lane[None, :] % $(2048))), dtype=tl.float32)
    r2_right = tl.load($(candidate_vmul_ptr) + (((r2_col[None, :] % $(384)) * $(2048)) + (r2_lane[:, None] % $(2048))), dtype=tl.float32)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value.to(tl.float32)
    tl.store($(candidate_vmm_ptr) + ((r2_row[:, None] * $(384)) + r2_col[None, :]), r2_value)
    r3_z = tl.arange($(49152))
    r3_value = (tl.load($(candidate_vmm_ptr) + ((((r3_z / $(384)) % $(128)) * $(384)) + (r3_z % $(384))), dtype=tl.float32) + tl.load($(candidate_vmul_1_ptr) + (r3_z % $(384)), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(49152)])
    tl.store($(candidate_vadd_ptr) + r3_z, r3_value)
    z = tl.arange($(49152))
    value = tl.load($(candidate_vadd_ptr) + ((((z / $(384)) % $(128)) * $(384)) + (z % $(384))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(49152)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_triton_router_matmul_production
