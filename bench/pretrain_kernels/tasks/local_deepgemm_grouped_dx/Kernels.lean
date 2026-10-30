import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_dx
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
  let vmm_2_ptr : RegionName := "vmm_2"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_ptr : RegionName := "vadd"
  triton {
    r0_z = tl.arange($(2621440))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(1310720)) % $(2)) * $(1310720)) + (((r0_z / $(2048)) % $(640)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(2621440)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(128))
    r1_col = tl.arange($(2048))
    r1_lane = tl.arange($(640))
    r1_left = tl.load($(input0_ptr) + ((((r1_row[:, None] + $(128)) % $(256)) * $(640)) + (r1_lane[None, :] % $(640))), dtype=tl.bfloat16)
    r1_right = tl.load($(vmul_ptr) + (((r1_lane[:, None] % $(640)) * $(2048)) + (r1_col[None, :] % $(2048)) + $(1310720)), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r1_row[:, None] * $(2048)) + r1_col[None, :]), r1_value)
    r2_z = tl.arange($(524288))
    r2_value = tl.where(((((r2_z / $(2048)) % $(256)) >= $(128)) & (((r2_z / $(2048)) % $(256)) < $(256)) & ((((r2_z / $(2048)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(vmm_2_ptr) + ((((((r2_z / $(2048)) % $(256)) - $(128)) % $(128)) * $(2048)) + (r2_z % $(2048))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(524288)])
    tl.store($(vslice_backward_ptr) + r2_z, r2_value)
    r3_row = tl.arange($(128))
    r3_col = tl.arange($(2048))
    r3_lane = tl.arange($(640))
    r3_left = tl.load($(input0_ptr) + (((r3_row[:, None] % $(256)) * $(640)) + (r3_lane[None, :] % $(640))), dtype=tl.bfloat16)
    r3_right = tl.load($(vmul_ptr) + (((r3_lane[:, None] % $(640)) * $(2048)) + (r3_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r3_value = tl.dot(r3_left, r3_right)
    r3_value = r3_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r3_row[:, None] * $(2048)) + r3_col[None, :]), r3_value)
    r4_z = tl.arange($(524288))
    r4_value = tl.where(((((r4_z / $(2048)) % $(256)) >= $(0)) & (((r4_z / $(2048)) % $(256)) < $(128)) & ((((r4_z / $(2048)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(vmm_3_ptr) + ((((((r4_z / $(2048)) % $(256)) - $(0)) % $(128)) * $(2048)) + (r4_z % $(2048))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(524288)])
    tl.store($(vslice_backward_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(524288))
    r5_value = (tl.load($(vslice_backward_ptr) + ((((r5_z / $(2048)) % $(256)) * $(2048)) + (r5_z % $(2048))), dtype=tl.bfloat16) + tl.load($(vslice_backward_1_ptr) + ((((r5_z / $(2048)) % $(256)) * $(2048)) + (r5_z % $(2048))), dtype=tl.bfloat16))
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(524288)])
    tl.store($(vadd_ptr) + r5_z, r5_value)
    z = tl.arange($(524288))
    value = tl.load($(vadd_ptr) + ((((z / $(2048)) % $(256)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vmm_3_ptr : RegionName := "vmm_3"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vcat_1_ptr : RegionName := "candidate_vcat_1"
  triton {
    r0_z = tl.arange($(2621440))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(1310720)) % $(2)) * $(1310720)) + (((r0_z / $(2048)) % $(640)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(2621440)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_row = tl.arange($(128))
    r1_col = tl.arange($(2048))
    r1_lane = tl.arange($(640))
    r1_left = tl.load($(input0_ptr) + ((((((r1_row[:, None] * $(640)) + r1_lane[None, :]) / $(640)) % $(256)) * $(640)) + (((r1_row[:, None] * $(640)) + r1_lane[None, :]) % $(640))), dtype=tl.bfloat16)
    r1_right = tl.load($(candidate_vmul_ptr) + (((r1_lane[:, None] % $(640)) * $(2048)) + (r1_col[None, :] % $(2048))), dtype=tl.bfloat16)
    r1_value = tl.dot(r1_left, r1_right)
    r1_value = r1_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r1_row[:, None] * $(2048)) + r1_col[None, :]), r1_value)
    r2_row = tl.arange($(128))
    r2_col = tl.arange($(2048))
    r2_lane = tl.arange($(640))
    r2_left = tl.load($(input0_ptr) + ((((((r2_row[:, None] * $(640)) + r2_lane[None, :] + $(81920)) / $(640)) % $(256)) * $(640)) + (((r2_row[:, None] * $(640)) + r2_lane[None, :] + $(81920)) % $(640))), dtype=tl.bfloat16)
    r2_right = tl.load($(candidate_vmul_ptr) + (((r2_lane[:, None] % $(640)) * $(2048)) + (r2_col[None, :] % $(2048)) + $(1310720)), dtype=tl.bfloat16)
    r2_value = tl.dot(r2_left, r2_right)
    r2_value = r2_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_3_ptr) + ((r2_row[:, None] * $(2048)) + r2_col[None, :]), r2_value)
    r3_z = tl.arange($(524288))
    r3_value = tl.where(((r3_z / $(2048)) % $(256)) < $(128), tl.load($(candidate_vmm_2_ptr) + ((((((r3_z / $(2048)) % $(256)) - $(0)) % $(128)) * $(2048)) + (r3_z % $(2048))), dtype=tl.bfloat16), tl.load($(candidate_vmm_3_ptr) + ((((((r3_z / $(2048)) % $(256)) - $(128)) % $(128)) * $(2048)) + (r3_z % $(2048))), dtype=tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(524288)])
    tl.store($(candidate_vcat_1_ptr) + r3_z, r3_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vcat_1_ptr) + ((((z / $(2048)) % $(256)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_dx
