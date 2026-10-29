import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_situ_glu_b30x30_sigmoid_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vdiv_ptr : RegionName := "vdiv"
  let vtanh_ptr : RegionName := "vtanh"
  let vmul_ptr : RegionName := "vmul"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vtanh_1_ptr : RegionName := "vtanh_1"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  triton {
    r0_z = tl.arange($(1572864))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(786432)) % $(2)) * $(1572864)) + (((r0_z / $(6144)) % $(128)) * $(12288)) + (r0_z % $(6144))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(1572864)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1572864))
    r1_value = tl.load($(input0_ptr) + ((((r1_z / $(786432)) % $(2)) * $(1572864)) + (((r1_z / $(6144)) % $(128)) * $(12288)) + (r1_z % $(6144)) + $(6144)), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(1572864)])
    tl.store($(vto_copy_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(1572864))
    r2_value = (tl.load($(vto_copy_ptr) + ((((r2_z / $(786432)) % $(2)) * $(786432)) + (((r2_z / $(6144)) % $(128)) * $(6144)) + (r2_z % $(6144))), dtype=tl.float32) / 30.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(1572864)])
    tl.store($(vdiv_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1572864))
    r3_value = tl.tanh(tl.load($(vdiv_ptr) + ((((r3_z / $(786432)) % $(2)) * $(786432)) + (((r3_z / $(6144)) % $(128)) * $(6144)) + (r3_z % $(6144))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(1572864)])
    tl.store($(vtanh_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(1572864))
    r4_value = (tl.load($(vtanh_ptr) + ((((r4_z / $(786432)) % $(2)) * $(786432)) + (((r4_z / $(6144)) % $(128)) * $(6144)) + (r4_z % $(6144))), dtype=tl.float32) * 30.0)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(1572864)])
    tl.store($(vmul_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(1572864))
    r5_value = tl.sigmoid(tl.load($(vto_copy_ptr) + ((((r5_z / $(786432)) % $(2)) * $(786432)) + (((r5_z / $(6144)) % $(128)) * $(6144)) + (r5_z % $(6144))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(1572864)])
    tl.store($(vsigmoid_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(1572864))
    r6_value = (tl.load($(vmul_ptr) + ((((r6_z / $(786432)) % $(2)) * $(786432)) + (((r6_z / $(6144)) % $(128)) * $(6144)) + (r6_z % $(6144))), dtype=tl.float32) * tl.load($(vsigmoid_ptr) + ((((r6_z / $(786432)) % $(2)) * $(786432)) + (((r6_z / $(6144)) % $(128)) * $(6144)) + (r6_z % $(6144))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(1572864)])
    tl.store($(vmul_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1572864))
    r7_value = (tl.load($(vto_copy_1_ptr) + ((((r7_z / $(786432)) % $(2)) * $(786432)) + (((r7_z / $(6144)) % $(128)) * $(6144)) + (r7_z % $(6144))), dtype=tl.float32) / 30.0)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1572864)])
    tl.store($(vdiv_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(1572864))
    r8_value = tl.tanh(tl.load($(vdiv_1_ptr) + ((((r8_z / $(786432)) % $(2)) * $(786432)) + (((r8_z / $(6144)) % $(128)) * $(6144)) + (r8_z % $(6144))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(1572864)])
    tl.store($(vtanh_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(1572864))
    r9_value = (tl.load($(vtanh_1_ptr) + ((((r9_z / $(786432)) % $(2)) * $(786432)) + (((r9_z / $(6144)) % $(128)) * $(6144)) + (r9_z % $(6144))), dtype=tl.float32) * 30.0)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(1572864)])
    tl.store($(vmul_2_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(1572864))
    r10_value = (tl.load($(vmul_1_ptr) + ((((r10_z / $(786432)) % $(2)) * $(786432)) + (((r10_z / $(6144)) % $(128)) * $(6144)) + (r10_z % $(6144))), dtype=tl.float32) * tl.load($(vmul_2_ptr) + ((((r10_z / $(786432)) % $(2)) * $(786432)) + (((r10_z / $(6144)) % $(128)) * $(6144)) + (r10_z % $(6144))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1572864)])
    tl.store($(vmul_3_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1572864))
    r11_value = tl.load($(vmul_3_ptr) + ((((r11_z / $(786432)) % $(2)) * $(786432)) + (((r11_z / $(6144)) % $(128)) * $(6144)) + (r11_z % $(6144))), dtype=tl.float32)
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(1572864)])
    tl.store($(vto_copy_2_ptr) + r11_z, r11_value)
    z = tl.arange($(1572864))
    value = tl.load($(vto_copy_2_ptr) + ((((z / $(786432)) % $(2)) * $(786432)) + (((z / $(6144)) % $(128)) * $(6144)) + (z % $(6144))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(1572864)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vdiv_ptr : RegionName := "vdiv"
  let vtanh_ptr : RegionName := "vtanh"
  let vmul_ptr : RegionName := "vmul"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vtanh_1_ptr : RegionName := "vtanh_1"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vtanh_ptr : RegionName := "candidate_vtanh"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vsigmoid_ptr : RegionName := "candidate_vsigmoid"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_vtanh_1_ptr : RegionName := "candidate_vtanh_1"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  triton {
    r0_z = tl.arange($(1572864))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(786432)) % $(2)) * $(1572864)) + (((r0_z / $(6144)) % $(128)) * $(12288)) + (r0_z % $(6144))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(1572864)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(1572864))
    r1_value = tl.load($(input0_ptr) + ((((r1_z / $(786432)) % $(2)) * $(1572864)) + (((r1_z / $(6144)) % $(128)) * $(12288)) + (r1_z % $(6144)) + $(6144)), dtype=tl.bfloat16)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(1572864)])
    tl.store($(candidate_vto_copy_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(1572864))
    r2_value = (tl.load($(candidate_vto_copy_ptr) + ((((r2_z / $(786432)) % $(2)) * $(786432)) + (((r2_z / $(6144)) % $(128)) * $(6144)) + (r2_z % $(6144))), dtype=tl.float32) / 30.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(1572864)])
    tl.store($(candidate_vdiv_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(1572864))
    r3_value = tl.tanh(tl.load($(candidate_vdiv_ptr) + ((((r3_z / $(786432)) % $(2)) * $(786432)) + (((r3_z / $(6144)) % $(128)) * $(6144)) + (r3_z % $(6144))), dtype=tl.float32))
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(1572864)])
    tl.store($(candidate_vtanh_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(1572864))
    r4_value = (tl.load($(candidate_vtanh_ptr) + ((((r4_z / $(786432)) % $(2)) * $(786432)) + (((r4_z / $(6144)) % $(128)) * $(6144)) + (r4_z % $(6144))), dtype=tl.float32) * 30.0)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(1572864)])
    tl.store($(candidate_vmul_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(1572864))
    r5_value = tl.sigmoid(tl.load($(candidate_vto_copy_ptr) + ((((r5_z / $(786432)) % $(2)) * $(786432)) + (((r5_z / $(6144)) % $(128)) * $(6144)) + (r5_z % $(6144))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(1572864)])
    tl.store($(candidate_vsigmoid_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(1572864))
    r6_value = (tl.load($(candidate_vmul_ptr) + ((((r6_z / $(786432)) % $(2)) * $(786432)) + (((r6_z / $(6144)) % $(128)) * $(6144)) + (r6_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vsigmoid_ptr) + ((((r6_z / $(786432)) % $(2)) * $(786432)) + (((r6_z / $(6144)) % $(128)) * $(6144)) + (r6_z % $(6144))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(1572864)])
    tl.store($(candidate_vmul_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1572864))
    r7_value = (tl.load($(candidate_vto_copy_1_ptr) + ((((r7_z / $(786432)) % $(2)) * $(786432)) + (((r7_z / $(6144)) % $(128)) * $(6144)) + (r7_z % $(6144))), dtype=tl.float32) / 30.0)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1572864)])
    tl.store($(candidate_vdiv_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(1572864))
    r8_value = tl.tanh(tl.load($(candidate_vdiv_1_ptr) + ((((r8_z / $(786432)) % $(2)) * $(786432)) + (((r8_z / $(6144)) % $(128)) * $(6144)) + (r8_z % $(6144))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(1572864)])
    tl.store($(candidate_vtanh_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(1572864))
    r9_value = (tl.load($(candidate_vtanh_1_ptr) + ((((r9_z / $(786432)) % $(2)) * $(786432)) + (((r9_z / $(6144)) % $(128)) * $(6144)) + (r9_z % $(6144))), dtype=tl.float32) * 30.0)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(1572864)])
    tl.store($(candidate_vmul_2_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(1572864))
    r10_value = (tl.load($(candidate_vmul_1_ptr) + ((((r10_z / $(786432)) % $(2)) * $(786432)) + (((r10_z / $(6144)) % $(128)) * $(6144)) + (r10_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vmul_2_ptr) + ((((r10_z / $(786432)) % $(2)) * $(786432)) + (((r10_z / $(6144)) % $(128)) * $(6144)) + (r10_z % $(6144))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1572864)])
    tl.store($(candidate_vmul_3_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1572864))
    r11_value = tl.load($(candidate_vmul_3_ptr) + ((((r11_z / $(786432)) % $(2)) * $(786432)) + (((r11_z / $(6144)) % $(128)) * $(6144)) + (r11_z % $(6144))), dtype=tl.float32)
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(1572864)])
    tl.store($(candidate_vto_copy_2_ptr) + r11_z, r11_value)
    z = tl.arange($(1572864))
    value = tl.load($(candidate_vto_copy_2_ptr) + ((((z / $(786432)) % $(2)) * $(786432)) + (((z / $(6144)) % $(128)) * $(6144)) + (z % $(6144))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(1572864)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_situ_glu_b30x30_sigmoid_fwd
