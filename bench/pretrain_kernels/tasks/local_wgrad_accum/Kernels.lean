import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_wgrad_accum
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmm_ptr : RegionName := "vmm"
  let vadd_ptr : RegionName := "vadd"
  triton {
    r0_z = tl.arange($(12582912))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(6144)) * $(2048)) + (r0_z % $(2048))), dtype=tl.float32) * 16.0)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(12582912)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(6144))
    r1_col = tl.arange($(2048))
    r1_lane = tl.arange($(256))
    r1_left = tl.load($(input1_ptr) + (((r1_lane[None, :] % $(256)) * $(6144)) + (r1_row[:, None] % $(6144))), dtype=tl.bfloat16)
    r1_right = tl.load($(input0_ptr) + (((r1_lane[:, None] % $(256)) * $(2048)) + (r1_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.float32)
    tl.store($(vmm_ptr) + ((r1_row[:, None] * $(2048)) + r1_col[None, :]), r1_value)
    r2_z = tl.arange($(12582912))
    r2_value = (tl.load($(vmul_ptr) + ((((r2_z / $(2048)) % $(6144)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32) + tl.load($(vmm_ptr) + ((((r2_z / $(2048)) % $(6144)) * $(2048)) + (r2_z % $(2048))), dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(12582912)])
    tl.store($(vadd_ptr) + r2_z, r2_value)
    z = tl.arange($(12582912))
    value = tl.load($(vadd_ptr) + ((((z / $(2048)) % $(6144)) * $(2048)) + (z % $(2048))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(12582912)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vaddmm_ptr : RegionName := "candidate_vaddmm"
  triton {
    r0_z = tl.arange($(12582912))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(6144)) * $(2048)) + (r0_z % $(2048))), dtype=tl.float32) * 16.0)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(12582912)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(6144))
    r1_col = tl.arange($(2048))
    r1_lane = tl.arange($(256))
    r1_left = tl.load($(input1_ptr) + (((r1_lane[None, :] % $(256)) * $(6144)) + (r1_row[:, None] % $(6144))), dtype=tl.bfloat16)
    r1_right = tl.load($(input0_ptr) + (((r1_lane[:, None] % $(256)) * $(2048)) + (r1_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value + tl.load($(candidate_vmul_ptr) + (((r1_row[:, None] % $(6144)) * $(2048)) + (r1_col[None, :] % $(2048))), dtype=tl.float32) * 1.0
    r1_value = r1_value.to(tl.float32)
    tl.store($(candidate_vaddmm_ptr) + ((r1_row[:, None] * $(2048)) + r1_col[None, :]), r1_value)
    z = tl.arange($(12582912))
    value = tl.load($(candidate_vaddmm_ptr) + ((((z / $(2048)) % $(6144)) * $(2048)) + (z % $(2048))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(12582912)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_wgrad_accum
