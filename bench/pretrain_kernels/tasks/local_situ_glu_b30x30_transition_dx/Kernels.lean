import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_situ_glu_b30x30_transition_dx
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
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vtanh_backward_ptr : RegionName := "vtanh_backward"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vtanh_backward_1_ptr : RegionName := "vtanh_backward_1"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vcat_ptr : RegionName := "vcat"
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
    r10_value = tl.load($(input1_ptr) + ((((r10_z / $(786432)) % $(2)) * $(786432)) + (((r10_z / $(6144)) % $(128)) * $(6144)) + (r10_z % $(6144))), dtype=tl.bfloat16)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1572864)])
    tl.store($(vto_copy_3_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1572864))
    r11_value = (tl.load($(vto_copy_3_ptr) + ((((r11_z / $(786432)) % $(2)) * $(786432)) + (((r11_z / $(6144)) % $(128)) * $(6144)) + (r11_z % $(6144))), dtype=tl.float32) * tl.load($(vmul_1_ptr) + ((((r11_z / $(786432)) % $(2)) * $(786432)) + (((r11_z / $(6144)) % $(128)) * $(6144)) + (r11_z % $(6144))), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(1572864)])
    tl.store($(vmul_4_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(1572864))
    r12_value = (tl.load($(vto_copy_3_ptr) + ((((r12_z / $(786432)) % $(2)) * $(786432)) + (((r12_z / $(6144)) % $(128)) * $(6144)) + (r12_z % $(6144))), dtype=tl.float32) * tl.load($(vmul_2_ptr) + ((((r12_z / $(786432)) % $(2)) * $(786432)) + (((r12_z / $(6144)) % $(128)) * $(6144)) + (r12_z % $(6144))), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(1572864)])
    tl.store($(vmul_5_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(1572864))
    r13_value = (tl.load($(vmul_4_ptr) + ((((r13_z / $(786432)) % $(2)) * $(786432)) + (((r13_z / $(6144)) % $(128)) * $(6144)) + (r13_z % $(6144))), dtype=tl.float32) * 30.0)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(1572864)])
    tl.store($(vmul_6_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(1572864))
    r14_value = (tl.load($(vmul_6_ptr) + ((((r14_z / $(786432)) % $(2)) * $(786432)) + (((r14_z / $(6144)) % $(128)) * $(6144)) + (r14_z % $(6144))), dtype=tl.float32) * ($(1) - tl.load($(vtanh_1_ptr) + ((((r14_z / $(786432)) % $(2)) * $(786432)) + (((r14_z / $(6144)) % $(128)) * $(6144)) + (r14_z % $(6144))), dtype=tl.float32) * tl.load($(vtanh_1_ptr) + ((((r14_z / $(786432)) % $(2)) * $(786432)) + (((r14_z / $(6144)) % $(128)) * $(6144)) + (r14_z % $(6144))), dtype=tl.float32)))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(1572864)])
    tl.store($(vtanh_backward_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(1572864))
    r15_value = (tl.load($(vtanh_backward_ptr) + ((((r15_z / $(786432)) % $(2)) * $(786432)) + (((r15_z / $(6144)) % $(128)) * $(6144)) + (r15_z % $(6144))), dtype=tl.float32) / 30.0)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(1572864)])
    tl.store($(vdiv_2_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(1572864))
    r16_value = (tl.load($(vmul_5_ptr) + ((((r16_z / $(786432)) % $(2)) * $(786432)) + (((r16_z / $(6144)) % $(128)) * $(6144)) + (r16_z % $(6144))), dtype=tl.float32) * tl.load($(vmul_ptr) + ((((r16_z / $(786432)) % $(2)) * $(786432)) + (((r16_z / $(6144)) % $(128)) * $(6144)) + (r16_z % $(6144))), dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(1572864)])
    tl.store($(vmul_7_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(1572864))
    r17_value = (tl.load($(vmul_5_ptr) + ((((r17_z / $(786432)) % $(2)) * $(786432)) + (((r17_z / $(6144)) % $(128)) * $(6144)) + (r17_z % $(6144))), dtype=tl.float32) * tl.load($(vsigmoid_ptr) + ((((r17_z / $(786432)) % $(2)) * $(786432)) + (((r17_z / $(6144)) % $(128)) * $(6144)) + (r17_z % $(6144))), dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(1572864)])
    tl.store($(vmul_8_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(1572864))
    r18_value = (tl.load($(vmul_7_ptr) + ((((r18_z / $(786432)) % $(2)) * $(786432)) + (((r18_z / $(6144)) % $(128)) * $(6144)) + (r18_z % $(6144))), dtype=tl.float32) * tl.load($(vsigmoid_ptr) + ((((r18_z / $(786432)) % $(2)) * $(786432)) + (((r18_z / $(6144)) % $(128)) * $(6144)) + (r18_z % $(6144))), dtype=tl.float32) * ($(1) - tl.load($(vsigmoid_ptr) + ((((r18_z / $(786432)) % $(2)) * $(786432)) + (((r18_z / $(6144)) % $(128)) * $(6144)) + (r18_z % $(6144))), dtype=tl.float32)))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(1572864)])
    tl.store($(vsigmoid_backward_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(1572864))
    r19_value = (tl.load($(vmul_8_ptr) + ((((r19_z / $(786432)) % $(2)) * $(786432)) + (((r19_z / $(6144)) % $(128)) * $(6144)) + (r19_z % $(6144))), dtype=tl.float32) * 30.0)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(1572864)])
    tl.store($(vmul_9_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(1572864))
    r20_value = (tl.load($(vmul_9_ptr) + ((((r20_z / $(786432)) % $(2)) * $(786432)) + (((r20_z / $(6144)) % $(128)) * $(6144)) + (r20_z % $(6144))), dtype=tl.float32) * ($(1) - tl.load($(vtanh_ptr) + ((((r20_z / $(786432)) % $(2)) * $(786432)) + (((r20_z / $(6144)) % $(128)) * $(6144)) + (r20_z % $(6144))), dtype=tl.float32) * tl.load($(vtanh_ptr) + ((((r20_z / $(786432)) % $(2)) * $(786432)) + (((r20_z / $(6144)) % $(128)) * $(6144)) + (r20_z % $(6144))), dtype=tl.float32)))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(1572864)])
    tl.store($(vtanh_backward_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(1572864))
    r21_value = (tl.load($(vtanh_backward_1_ptr) + ((((r21_z / $(786432)) % $(2)) * $(786432)) + (((r21_z / $(6144)) % $(128)) * $(6144)) + (r21_z % $(6144))), dtype=tl.float32) / 30.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(1572864)])
    tl.store($(vdiv_3_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(1572864))
    r22_value = (tl.load($(vsigmoid_backward_ptr) + ((((r22_z / $(786432)) % $(2)) * $(786432)) + (((r22_z / $(6144)) % $(128)) * $(6144)) + (r22_z % $(6144))), dtype=tl.float32) + tl.load($(vdiv_3_ptr) + ((((r22_z / $(786432)) % $(2)) * $(786432)) + (((r22_z / $(6144)) % $(128)) * $(6144)) + (r22_z % $(6144))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(1572864)])
    tl.store($(vadd_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(1572864))
    r23_value = tl.load($(vdiv_2_ptr) + ((((r23_z / $(786432)) % $(2)) * $(786432)) + (((r23_z / $(6144)) % $(128)) * $(6144)) + (r23_z % $(6144))), dtype=tl.float32)
    r23_value = r23_value.to(tl.bfloat16)
    r23_value = tl.broadcast_to(r23_value, [$(1572864)])
    tl.store($(vto_copy_4_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(1572864))
    r24_value = tl.load($(vadd_ptr) + ((((r24_z / $(786432)) % $(2)) * $(786432)) + (((r24_z / $(6144)) % $(128)) * $(6144)) + (r24_z % $(6144))), dtype=tl.float32)
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(1572864)])
    tl.store($(vto_copy_5_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(3145728))
    r25_value = tl.where((r25_z % $(12288)) < $(6144), tl.load($(vto_copy_5_ptr) + ((((r25_z / $(1572864)) % $(2)) * $(786432)) + (((r25_z / $(12288)) % $(128)) * $(6144)) + (((r25_z % $(12288)) - $(0)) % $(6144))), dtype=tl.bfloat16), tl.load($(vto_copy_4_ptr) + ((((r25_z / $(1572864)) % $(2)) * $(786432)) + (((r25_z / $(12288)) % $(128)) * $(6144)) + (((r25_z % $(12288)) - $(6144)) % $(6144))), dtype=tl.bfloat16))
    r25_value = r25_value.to(tl.bfloat16)
    r25_value = tl.broadcast_to(r25_value, [$(3145728)])
    tl.store($(vcat_ptr) + r25_z, r25_value)
    z = tl.arange($(3145728))
    value = tl.load($(vcat_ptr) + ((((z / $(1572864)) % $(2)) * $(1572864)) + (((z / $(12288)) % $(128)) * $(12288)) + (z % $(12288))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(3145728)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
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
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vtanh_backward_ptr : RegionName := "vtanh_backward"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vtanh_backward_1_ptr : RegionName := "vtanh_backward_1"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vadd_ptr : RegionName := "vadd"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vcat_ptr : RegionName := "vcat"
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
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vmul_6_ptr : RegionName := "candidate_vmul_6"
  let candidate_vtanh_backward_ptr : RegionName := "candidate_vtanh_backward"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vsigmoid_backward_ptr : RegionName := "candidate_vsigmoid_backward"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vtanh_backward_1_ptr : RegionName := "candidate_vtanh_backward_1"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
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
    r10_value = tl.load($(input1_ptr) + ((((r10_z / $(786432)) % $(2)) * $(786432)) + (((r10_z / $(6144)) % $(128)) * $(6144)) + (r10_z % $(6144))), dtype=tl.bfloat16)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1572864)])
    tl.store($(candidate_vto_copy_3_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1572864))
    r11_value = (tl.load($(candidate_vto_copy_3_ptr) + ((((r11_z / $(786432)) % $(2)) * $(786432)) + (((r11_z / $(6144)) % $(128)) * $(6144)) + (r11_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vmul_1_ptr) + ((((r11_z / $(786432)) % $(2)) * $(786432)) + (((r11_z / $(6144)) % $(128)) * $(6144)) + (r11_z % $(6144))), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(1572864)])
    tl.store($(candidate_vmul_4_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(1572864))
    r12_value = (tl.load($(candidate_vto_copy_3_ptr) + ((((r12_z / $(786432)) % $(2)) * $(786432)) + (((r12_z / $(6144)) % $(128)) * $(6144)) + (r12_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vmul_2_ptr) + ((((r12_z / $(786432)) % $(2)) * $(786432)) + (((r12_z / $(6144)) % $(128)) * $(6144)) + (r12_z % $(6144))), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(1572864)])
    tl.store($(candidate_vmul_5_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(1572864))
    r13_value = (tl.load($(candidate_vmul_4_ptr) + ((((r13_z / $(786432)) % $(2)) * $(786432)) + (((r13_z / $(6144)) % $(128)) * $(6144)) + (r13_z % $(6144))), dtype=tl.float32) * 30.0)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(1572864)])
    tl.store($(candidate_vmul_6_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(1572864))
    r14_value = (tl.load($(candidate_vmul_6_ptr) + ((((r14_z / $(786432)) % $(2)) * $(786432)) + (((r14_z / $(6144)) % $(128)) * $(6144)) + (r14_z % $(6144))), dtype=tl.float32) * ($(1) - tl.load($(candidate_vtanh_1_ptr) + ((((r14_z / $(786432)) % $(2)) * $(786432)) + (((r14_z / $(6144)) % $(128)) * $(6144)) + (r14_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vtanh_1_ptr) + ((((r14_z / $(786432)) % $(2)) * $(786432)) + (((r14_z / $(6144)) % $(128)) * $(6144)) + (r14_z % $(6144))), dtype=tl.float32)))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(1572864)])
    tl.store($(candidate_vtanh_backward_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(1572864))
    r15_value = (tl.load($(candidate_vtanh_backward_ptr) + ((((r15_z / $(786432)) % $(2)) * $(786432)) + (((r15_z / $(6144)) % $(128)) * $(6144)) + (r15_z % $(6144))), dtype=tl.float32) / 30.0)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(1572864)])
    tl.store($(candidate_vdiv_2_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(1572864))
    r16_value = (tl.load($(candidate_vmul_5_ptr) + ((((r16_z / $(786432)) % $(2)) * $(786432)) + (((r16_z / $(6144)) % $(128)) * $(6144)) + (r16_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vmul_ptr) + ((((r16_z / $(786432)) % $(2)) * $(786432)) + (((r16_z / $(6144)) % $(128)) * $(6144)) + (r16_z % $(6144))), dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(1572864)])
    tl.store($(candidate_vmul_7_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(1572864))
    r17_value = (tl.load($(candidate_vmul_5_ptr) + ((((r17_z / $(786432)) % $(2)) * $(786432)) + (((r17_z / $(6144)) % $(128)) * $(6144)) + (r17_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vsigmoid_ptr) + ((((r17_z / $(786432)) % $(2)) * $(786432)) + (((r17_z / $(6144)) % $(128)) * $(6144)) + (r17_z % $(6144))), dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(1572864)])
    tl.store($(candidate_vmul_8_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(1572864))
    r18_value = (tl.load($(candidate_vmul_7_ptr) + ((((r18_z / $(786432)) % $(2)) * $(786432)) + (((r18_z / $(6144)) % $(128)) * $(6144)) + (r18_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vsigmoid_ptr) + ((((r18_z / $(786432)) % $(2)) * $(786432)) + (((r18_z / $(6144)) % $(128)) * $(6144)) + (r18_z % $(6144))), dtype=tl.float32) * ($(1) - tl.load($(candidate_vsigmoid_ptr) + ((((r18_z / $(786432)) % $(2)) * $(786432)) + (((r18_z / $(6144)) % $(128)) * $(6144)) + (r18_z % $(6144))), dtype=tl.float32)))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(1572864)])
    tl.store($(candidate_vsigmoid_backward_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(1572864))
    r19_value = (tl.load($(candidate_vmul_8_ptr) + ((((r19_z / $(786432)) % $(2)) * $(786432)) + (((r19_z / $(6144)) % $(128)) * $(6144)) + (r19_z % $(6144))), dtype=tl.float32) * 30.0)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(1572864)])
    tl.store($(candidate_vmul_9_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(1572864))
    r20_value = (tl.load($(candidate_vmul_9_ptr) + ((((r20_z / $(786432)) % $(2)) * $(786432)) + (((r20_z / $(6144)) % $(128)) * $(6144)) + (r20_z % $(6144))), dtype=tl.float32) * ($(1) - tl.load($(candidate_vtanh_ptr) + ((((r20_z / $(786432)) % $(2)) * $(786432)) + (((r20_z / $(6144)) % $(128)) * $(6144)) + (r20_z % $(6144))), dtype=tl.float32) * tl.load($(candidate_vtanh_ptr) + ((((r20_z / $(786432)) % $(2)) * $(786432)) + (((r20_z / $(6144)) % $(128)) * $(6144)) + (r20_z % $(6144))), dtype=tl.float32)))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(1572864)])
    tl.store($(candidate_vtanh_backward_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(1572864))
    r21_value = (tl.load($(candidate_vtanh_backward_1_ptr) + ((((r21_z / $(786432)) % $(2)) * $(786432)) + (((r21_z / $(6144)) % $(128)) * $(6144)) + (r21_z % $(6144))), dtype=tl.float32) / 30.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(1572864)])
    tl.store($(candidate_vdiv_3_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(1572864))
    r22_value = (tl.load($(candidate_vsigmoid_backward_ptr) + ((((r22_z / $(786432)) % $(2)) * $(786432)) + (((r22_z / $(6144)) % $(128)) * $(6144)) + (r22_z % $(6144))), dtype=tl.float32) + tl.load($(candidate_vdiv_3_ptr) + ((((r22_z / $(786432)) % $(2)) * $(786432)) + (((r22_z / $(6144)) % $(128)) * $(6144)) + (r22_z % $(6144))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(1572864)])
    tl.store($(candidate_vadd_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(1572864))
    r23_value = tl.load($(candidate_vdiv_2_ptr) + ((((r23_z / $(786432)) % $(2)) * $(786432)) + (((r23_z / $(6144)) % $(128)) * $(6144)) + (r23_z % $(6144))), dtype=tl.float32)
    r23_value = r23_value.to(tl.bfloat16)
    r23_value = tl.broadcast_to(r23_value, [$(1572864)])
    tl.store($(candidate_vto_copy_4_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(1572864))
    r24_value = tl.load($(candidate_vadd_ptr) + ((((r24_z / $(786432)) % $(2)) * $(786432)) + (((r24_z / $(6144)) % $(128)) * $(6144)) + (r24_z % $(6144))), dtype=tl.float32)
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(1572864)])
    tl.store($(candidate_vto_copy_5_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(3145728))
    r25_value = tl.where((r25_z % $(12288)) < $(6144), tl.load($(candidate_vto_copy_5_ptr) + ((((r25_z / $(1572864)) % $(2)) * $(786432)) + (((r25_z / $(12288)) % $(128)) * $(6144)) + (((r25_z % $(12288)) - $(0)) % $(6144))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_4_ptr) + ((((r25_z / $(1572864)) % $(2)) * $(786432)) + (((r25_z / $(12288)) % $(128)) * $(6144)) + (((r25_z % $(12288)) - $(6144)) % $(6144))), dtype=tl.bfloat16))
    r25_value = r25_value.to(tl.bfloat16)
    r25_value = tl.broadcast_to(r25_value, [$(3145728)])
    tl.store($(candidate_vcat_ptr) + r25_z, r25_value)
    z = tl.arange($(3145728))
    value = tl.load($(candidate_vcat_ptr) + ((((z / $(1572864)) % $(2)) * $(1572864)) + (((z / $(12288)) % $(128)) * $(12288)) + (z % $(12288))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(3145728)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_situ_glu_b30x30_transition_dx
