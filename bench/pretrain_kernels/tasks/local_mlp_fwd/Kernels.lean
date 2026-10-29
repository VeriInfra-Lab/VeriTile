import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mlp_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let parameter2_ptr : RegionName := "parameter2"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmm_ptr : RegionName := "vmm"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsilu_ptr : RegionName := "vsilu"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmm_2_ptr : RegionName := "vmm_2"
  triton {
    r0_z = tl.arange($(12582912))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(6144)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(12582912)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(12582912))
    r1_value = (tl.load($(parameter1_ptr) + ((((r1_z / $(2048)) % $(6144)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(12582912)])
    tl.store($(vmul_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(12582912))
    r2_value = (tl.load($(parameter2_ptr) + ((((r2_z / $(6144)) % $(2048)) * $(6144)) + (r2_z % $(6144))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(12582912)])
    tl.store($(vmul_2_ptr) + r2_z, r2_value)
    r3_row = tl.arange($(256))
    r3_col = tl.arange($(6144))
    r3_lane = tl.arange($(2048))
    r3_left = tl.load($(input0_ptr) + ((((((r3_row[:, None] * $(2048)) + r3_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r3_row[:, None] * $(2048)) + r3_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r3_row[:, None] * $(2048)) + r3_lane[None, :]) % $(2048))), dtype=tl.bfloat16)
    r3_right = tl.load($(vmul_ptr) + (((r3_col[None, :] % $(6144)) * $(2048)) + (r3_lane[:, None] % $(2048))), dtype=tl.bfloat16)
    r3_value = tl.dot(r3_left, r3_right)
    r3_value = r3_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r3_row[:, None] * $(6144)) + r3_col[None, :]), r3_value)
    r4_row = tl.arange($(256))
    r4_col = tl.arange($(6144))
    r4_lane = tl.arange($(2048))
    r4_left = tl.load($(input0_ptr) + ((((((r4_row[:, None] * $(2048)) + r4_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r4_row[:, None] * $(2048)) + r4_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r4_row[:, None] * $(2048)) + r4_lane[None, :]) % $(2048))), dtype=tl.bfloat16)
    r4_right = tl.load($(vmul_1_ptr) + (((r4_col[None, :] % $(6144)) * $(2048)) + (r4_lane[:, None] % $(2048))), dtype=tl.bfloat16)
    r4_value = tl.dot(r4_left, r4_right)
    r4_value = r4_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r4_row[:, None] * $(6144)) + r4_col[None, :]), r4_value)
    r5_z = tl.arange($(1572864))
    r5_value = tl.load($(vmm_ptr) + ((((r5_z / $(786432)) % $(2)) * $(786432)) + (((r5_z / $(6144)) % $(128)) * $(6144)) + (r5_z % $(6144))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(1572864)])
    tl.store($(vto_copy_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(1572864))
    r6_value = (tl.load($(vto_copy_ptr) + ((((r6_z / $(786432)) % $(2)) * $(786432)) + (((r6_z / $(6144)) % $(128)) * $(6144)) + (r6_z % $(6144))), dtype=tl.float32) * tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r6_z / $(786432)) % $(2)) * $(786432)) + (((r6_z / $(6144)) % $(128)) * $(6144)) + (r6_z % $(6144))), dtype=tl.float32)))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(1572864)])
    tl.store($(vsilu_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1572864))
    r7_value = tl.load($(vmm_1_ptr) + ((((r7_z / $(786432)) % $(2)) * $(786432)) + (((r7_z / $(6144)) % $(128)) * $(6144)) + (r7_z % $(6144))), dtype=tl.bfloat16)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1572864)])
    tl.store($(vto_copy_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(1572864))
    r8_value = (tl.load($(vsilu_ptr) + ((((r8_z / $(786432)) % $(2)) * $(786432)) + (((r8_z / $(6144)) % $(128)) * $(6144)) + (r8_z % $(6144))), dtype=tl.float32) * tl.load($(vto_copy_1_ptr) + ((((r8_z / $(786432)) % $(2)) * $(786432)) + (((r8_z / $(6144)) % $(128)) * $(6144)) + (r8_z % $(6144))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(1572864)])
    tl.store($(vmul_3_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(1572864))
    r9_value = tl.load($(vmul_3_ptr) + ((((r9_z / $(786432)) % $(2)) * $(786432)) + (((r9_z / $(6144)) % $(128)) * $(6144)) + (r9_z % $(6144))), dtype=tl.float32)
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(1572864)])
    tl.store($(vto_copy_2_ptr) + r9_z, r9_value)
    r10_row = tl.arange($(256))
    r10_col = tl.arange($(2048))
    r10_lane = tl.arange($(6144))
    r10_left = tl.load($(vto_copy_2_ptr) + ((((((r10_row[:, None] * $(6144)) + r10_lane[None, :]) / $(786432)) % $(2)) * $(786432)) + (((((r10_row[:, None] * $(6144)) + r10_lane[None, :]) / $(6144)) % $(128)) * $(6144)) + (((r10_row[:, None] * $(6144)) + r10_lane[None, :]) % $(6144))), dtype=tl.bfloat16)
    r10_right = tl.load($(vmul_2_ptr) + (((r10_col[None, :] % $(2048)) * $(6144)) + (r10_lane[:, None] % $(6144))), dtype=tl.bfloat16)
    r10_value = tl.dot(r10_left, r10_right)
    r10_value = r10_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r10_row[:, None] * $(2048)) + r10_col[None, :]), r10_value)
    z = tl.arange($(524288))
    value = tl.load($(vmm_2_ptr) + ((((z / $(262144)) % $(2)) * $(262144)) + (((z / $(2048)) % $(128)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let parameter2_ptr : RegionName := "parameter2"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmm_ptr : RegionName := "vmm"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vsilu_ptr : RegionName := "vsilu"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vsilu_ptr : RegionName := "candidate_vsilu"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  triton {
    r0_z = tl.arange($(12582912))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(2048)) % $(6144)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(12582912)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(12582912))
    r1_value = (tl.load($(parameter1_ptr) + ((((r1_z / $(2048)) % $(6144)) * $(2048)) + (r1_z % $(2048))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(12582912)])
    tl.store($(candidate_vmul_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(12582912))
    r2_value = (tl.load($(parameter2_ptr) + ((((r2_z / $(6144)) % $(2048)) * $(6144)) + (r2_z % $(6144))), dtype=tl.bfloat16) * tl.cast(0.02, tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(12582912)])
    tl.store($(candidate_vmul_2_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(25165824))
    r3_value = tl.where(((r3_z / $(2048)) % $(12288)) < $(6144), tl.load($(candidate_vmul_ptr) + ((((((r3_z / $(2048)) % $(12288)) - $(0)) % $(6144)) * $(2048)) + (r3_z % $(2048))), dtype=tl.bfloat16), tl.load($(candidate_vmul_1_ptr) + ((((((r3_z / $(2048)) % $(12288)) - $(6144)) % $(6144)) * $(2048)) + (r3_z % $(2048))), dtype=tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(25165824)])
    tl.store($(candidate_vcat_ptr) + r3_z, r3_value)
    r4_row = tl.arange($(256))
    r4_col = tl.arange($(12288))
    r4_lane = tl.arange($(2048))
    r4_left = tl.load($(input0_ptr) + ((((((r4_row[:, None] * $(2048)) + r4_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r4_row[:, None] * $(2048)) + r4_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r4_row[:, None] * $(2048)) + r4_lane[None, :]) % $(2048))), dtype=tl.bfloat16)
    r4_right = tl.load($(candidate_vcat_ptr) + (((r4_col[None, :] % $(12288)) * $(2048)) + (r4_lane[:, None] % $(2048))), dtype=tl.bfloat16)
    r4_value = tl.dot(r4_left, r4_right)
    r4_value = r4_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r4_row[:, None] * $(12288)) + r4_col[None, :]), r4_value)
    r5_z = tl.arange($(1572864))
    r5_value = tl.load($(candidate_vmm_ptr) + ((((r5_z / $(786432)) % $(2)) * $(1572864)) + (((r5_z / $(6144)) % $(128)) * $(12288)) + (r5_z % $(6144))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(1572864)])
    tl.store($(candidate_vto_copy_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(1572864))
    r6_value = (tl.load($(candidate_vto_copy_ptr) + ((((r6_z / $(786432)) % $(2)) * $(786432)) + (((r6_z / $(6144)) % $(128)) * $(6144)) + (r6_z % $(6144))), dtype=tl.float32) * tl.sigmoid(tl.load($(candidate_vto_copy_ptr) + ((((r6_z / $(786432)) % $(2)) * $(786432)) + (((r6_z / $(6144)) % $(128)) * $(6144)) + (r6_z % $(6144))), dtype=tl.float32)))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(1572864)])
    tl.store($(candidate_vsilu_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1572864))
    r7_value = tl.load($(candidate_vmm_ptr) + ((((r7_z / $(786432)) % $(2)) * $(1572864)) + (((r7_z / $(6144)) % $(128)) * $(12288)) + (r7_z % $(6144)) + $(6144)), dtype=tl.bfloat16)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1572864)])
    tl.store($(candidate_vto_copy_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(1572864))
    r8_value = (tl.load($(candidate_vsilu_ptr) + ((((r8_z / $(786432)) % $(2)) * $(786432)) + (((r8_z / $(6144)) % $(128)) * $(6144)) + (r8_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vto_copy_1_ptr) + ((((r8_z / $(786432)) % $(2)) * $(786432)) + (((r8_z / $(6144)) % $(128)) * $(6144)) + (r8_z % $(6144))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(1572864)])
    tl.store($(candidate_vmul_3_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(1572864))
    r9_value = tl.load($(candidate_vmul_3_ptr) + ((((r9_z / $(786432)) % $(2)) * $(786432)) + (((r9_z / $(6144)) % $(128)) * $(6144)) + (r9_z % $(6144))), dtype=tl.float32)
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(1572864)])
    tl.store($(candidate_vto_copy_2_ptr) + r9_z, r9_value)
    r10_row = tl.arange($(256))
    r10_col = tl.arange($(2048))
    r10_lane = tl.arange($(6144))
    r10_left = tl.load($(candidate_vto_copy_2_ptr) + ((((((r10_row[:, None] * $(6144)) + r10_lane[None, :]) / $(786432)) % $(2)) * $(786432)) + (((((r10_row[:, None] * $(6144)) + r10_lane[None, :]) / $(6144)) % $(128)) * $(6144)) + (((r10_row[:, None] * $(6144)) + r10_lane[None, :]) % $(6144))), dtype=tl.bfloat16)
    r10_right = tl.load($(candidate_vmul_2_ptr) + (((r10_col[None, :] % $(2048)) * $(6144)) + (r10_lane[:, None] % $(6144))), dtype=tl.bfloat16)
    r10_value = tl.dot(r10_left, r10_right)
    r10_value = r10_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r10_row[:, None] * $(2048)) + r10_col[None, :]), r10_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vmm_1_ptr) + ((((z / $(262144)) % $(2)) * $(262144)) + (((z / $(2048)) % $(128)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mlp_fwd
