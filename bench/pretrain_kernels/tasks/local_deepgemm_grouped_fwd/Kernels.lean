import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_fwd
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
  let vmm_1_ptr : RegionName := "vmm_1"
  let vcat_ptr : RegionName := "vcat"
  triton {
    r0_z = tl.arange($(2621440))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(1310720)) % $(2)) * $(1310720)) + (((r0_z / $(2048)) % $(640)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(2621440)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(128))
    r1_col = tl.arange($(640))
    r1_lane = tl.arange($(2048))
    r1_left = tl.load($(input0_ptr) + (((r1_row[:, None] % $(256)) * $(2048)) + (r1_lane[None, :] % $(2048))), dtype=tl.bfloat16)
    r1_right = tl.load($(vmul_ptr) + (((r1_col[None, :] % $(640)) * $(2048)) + (r1_lane[:, None] % $(2048))), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r1_row[:, None] * $(640)) + r1_col[None, :]), r1_value)
    r2_row = tl.arange($(128))
    r2_col = tl.arange($(640))
    r2_lane = tl.arange($(2048))
    r2_left = tl.load($(input0_ptr) + ((((r2_row[:, None] + $(128)) % $(256)) * $(2048)) + (r2_lane[None, :] % $(2048))), dtype=tl.bfloat16)
    r2_right = tl.load($(vmul_ptr) + (((r2_col[None, :] % $(640)) * $(2048)) + (r2_lane[:, None] % $(2048)) + $(1310720)), dtype=tl.bfloat16)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r2_row[:, None] * $(640)) + r2_col[None, :]), r2_value)
    r3_z = tl.arange($(163840))
    r3_value = tl.where(((r3_z / $(640)) % $(256)) < $(128), tl.load($(vmm_ptr) + ((((((r3_z / $(640)) % $(256)) - $(0)) % $(128)) * $(640)) + (r3_z % $(640))), dtype=tl.bfloat16), tl.load($(vmm_1_ptr) + ((((((r3_z / $(640)) % $(256)) - $(128)) % $(128)) * $(640)) + (r3_z % $(640))), dtype=tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(163840)])
    tl.store($(vcat_ptr) + r3_z, r3_value)
    z = tl.arange($(163840))
    value = tl.load($(vcat_ptr) + ((((z / $(640)) % $(256)) * $(640)) + (z % $(640))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(163840)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmm_ptr : RegionName := "vmm"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vcat_ptr : RegionName := "vcat"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  triton {
    r0_z = tl.arange($(2621440))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(1310720)) % $(2)) * $(1310720)) + (((r0_z / $(2048)) % $(640)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(2621440)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(128))
    r1_col = tl.arange($(640))
    r1_lane = tl.arange($(2048))
    r1_left = tl.load($(input0_ptr) + (((r1_row[:, None] % $(256)) * $(2048)) + (r1_lane[None, :] % $(2048))), dtype=tl.bfloat16)
    r1_right = tl.load($(candidate_vmul_ptr) + (((r1_col[None, :] % $(640)) * $(2048)) + (r1_lane[:, None] % $(2048))), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r1_row[:, None] * $(640)) + r1_col[None, :]), r1_value)
    r2_row = tl.arange($(128))
    r2_col = tl.arange($(640))
    r2_lane = tl.arange($(2048))
    r2_left = tl.load($(input0_ptr) + ((((r2_row[:, None] + $(128)) % $(256)) * $(2048)) + (r2_lane[None, :] % $(2048))), dtype=tl.bfloat16)
    r2_right = tl.load($(candidate_vmul_ptr) + (((r2_col[None, :] % $(640)) * $(2048)) + (r2_lane[:, None] % $(2048)) + $(1310720)), dtype=tl.bfloat16)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r2_row[:, None] * $(640)) + r2_col[None, :]), r2_value)
    r3_z = tl.arange($(163840))
    r3_value = tl.where(((r3_z / $(640)) % $(256)) < $(128), tl.load($(candidate_vmm_ptr) + ((((((r3_z / $(640)) % $(256)) - $(0)) % $(128)) * $(640)) + (r3_z % $(640))), dtype=tl.bfloat16), tl.load($(candidate_vmm_1_ptr) + ((((((r3_z / $(640)) % $(256)) - $(128)) % $(128)) * $(640)) + (r3_z % $(640))), dtype=tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(163840)])
    tl.store($(candidate_vcat_ptr) + r3_z, r3_value)
    z = tl.arange($(163840))
    value = tl.load($(candidate_vcat_ptr) + ((((z / $(640)) % $(256)) * $(640)) + (z % $(640))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(163840)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_fwd
