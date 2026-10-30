import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_rope_dx
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
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vbmm_ptr : RegionName := "vbmm"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcos_ptr : RegionName := "vcos"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vsin_ptr : RegionName := "vsin"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vneg_2_ptr : RegionName := "vneg_2"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vadd_3_ptr : RegionName := "vadd_3"
  triton {
    r0_z = tl.arange($(64))
    r0_value = tl.toReal((tl.cast(r0_z, tl.int64) * tl.full([], $(2), dtype=tl.int64)))
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(64)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(64))
    r1_value = (tl.load($(vto_copy_ptr) + r1_z, dtype=tl.float32) / 128.0)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(64)])
    tl.store($(vdiv_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(64))
    r2_value = tl.extra.cuda.libdevice.pow($(10000), tl.load($(vdiv_ptr) + r2_z, dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(64)])
    tl.store($(vpow_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(64))
    r3_value = 1.0 / tl.load($(vpow_1_ptr) + r3_z, dtype=tl.float32)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(64)])
    tl.store($(vreciprocal_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(64))
    r4_value = (tl.load($(vreciprocal_ptr) + r4_z, dtype=tl.float32) * 1.0)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(64)])
    tl.store($(vmul_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(128))
    r5_value = tl.toReal(tl.cast(r5_z, tl.int64))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(128)])
    tl.store($(vto_copy_2_ptr) + r5_z, r5_value)
    r6_row = tl.arange($(64))
    r6_col = tl.arange($(128))
    r6_lane = tl.arange($(1))
    for r6_group in range($(1)) {
      r6_left = tl.load($(vmul_ptr) + ((r6_row[:, None] + r6_lane[None, :]) % $(64)), dtype=tl.float32)
      r6_right = tl.load($(vto_copy_2_ptr) + (((r6_lane[:, None] * $(128)) + r6_col[None, :]) % $(128)), dtype=tl.float32)
      r6_value = tl.dot(r6_left, r6_right)
      r6_value = r6_value.to(tl.float32)
      tl.store($(vbmm_ptr) + ((r6_group * $(8192)) + (r6_row[:, None] * $(128)) + r6_col[None, :]), r6_value)
    }
    r7_z = tl.arange($(16384))
    r7_value = tl.where((r7_z % $(128)) < $(64), tl.load($(vbmm_ptr) + (((((r7_z % $(128)) - $(0)) % $(64)) * $(128)) + ((r7_z / $(128)) % $(128))), dtype=tl.float32), tl.load($(vbmm_ptr) + (((((r7_z % $(128)) - $(64)) % $(64)) * $(128)) + ((r7_z / $(128)) % $(128))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(16384)])
    tl.store($(vcat_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(16384))
    r8_value = tl.cos(tl.load($(vcat_1_ptr) + ((((r8_z / $(128)) % $(128)) * $(128)) + (r8_z % $(128))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(16384)])
    tl.store($(vcos_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(16384))
    r9_value = (tl.load($(vcos_ptr) + ((((r9_z / $(128)) % $(128)) * $(128)) + (r9_z % $(128))), dtype=tl.float32) * 1.0)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(16384)])
    tl.store($(vmul_3_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(16384))
    r10_value = tl.sin(tl.load($(vcat_1_ptr) + ((((r10_z / $(128)) % $(128)) * $(128)) + (r10_z % $(128))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(16384)])
    tl.store($(vsin_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(16384))
    r11_value = (tl.load($(vsin_ptr) + ((((r11_z / $(128)) % $(128)) * $(128)) + (r11_z % $(128))), dtype=tl.float32) * 1.0)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(16384)])
    tl.store($(vmul_4_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(16384))
    r12_value = tl.load($(vmul_3_ptr) + ((((r12_z / $(128)) % $(128)) * $(128)) + (r12_z % $(128))), dtype=tl.float32)
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(16384)])
    tl.store($(vto_copy_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(16384))
    r13_value = tl.load($(vmul_4_ptr) + ((((r13_z / $(128)) % $(128)) * $(128)) + (r13_z % $(128))), dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(16384)])
    tl.store($(vto_copy_4_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(524288))
    r14_value = (tl.load($(input0_ptr) + ((((r14_z / $(262144)) % $(2)) * $(262144)) + (((r14_z / $(16384)) % $(16)) * $(16384)) + (((r14_z / $(128)) % $(128)) * $(128)) + (r14_z % $(128))), dtype=tl.bfloat16) * tl.load($(vto_copy_4_ptr) + ((((r14_z / $(128)) % $(128)) * $(128)) + (r14_z % $(128))), dtype=tl.bfloat16))
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(524288)])
    tl.store($(vmul_9_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(262144))
    r15_value = -tl.load($(vmul_9_ptr) + ((((r15_z / $(131072)) % $(2)) * $(262144)) + (((r15_z / $(8192)) % $(16)) * $(16384)) + (((r15_z / $(64)) % $(128)) * $(128)) + (r15_z % $(64))), dtype=tl.bfloat16)
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(262144)])
    tl.store($(vneg_2_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(524288))
    r16_value = tl.where((((r16_z % $(128)) >= $(64)) & ((r16_z % $(128)) < $(128)) & (((r16_z % $(128)) - $(64)) % $(1) == $(0))), tl.load($(vneg_2_ptr) + ((((r16_z / $(262144)) % $(2)) * $(131072)) + (((r16_z / $(16384)) % $(16)) * $(8192)) + (((r16_z / $(128)) % $(128)) * $(64)) + (((r16_z % $(128)) - $(64)) % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r16_value = r16_value.to(tl.bfloat16)
    r16_value = tl.broadcast_to(r16_value, [$(524288)])
    tl.store($(vslice_backward_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(524288))
    r17_value = tl.where((((r17_z % $(128)) >= $(0)) & ((r17_z % $(128)) < $(64)) & (((r17_z % $(128)) - $(0)) % $(1) == $(0))), tl.load($(vmul_9_ptr) + ((((r17_z / $(262144)) % $(2)) * $(262144)) + (((r17_z / $(16384)) % $(16)) * $(16384)) + (((r17_z / $(128)) % $(128)) * $(128)) + ((((r17_z % $(128)) - $(0)) + $(64)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(524288)])
    tl.store($(vslice_backward_1_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(524288))
    r18_value = (tl.load($(vslice_backward_ptr) + ((((r18_z / $(262144)) % $(2)) * $(262144)) + (((r18_z / $(16384)) % $(16)) * $(16384)) + (((r18_z / $(128)) % $(128)) * $(128)) + (r18_z % $(128))), dtype=tl.bfloat16) + tl.load($(vslice_backward_1_ptr) + ((((r18_z / $(262144)) % $(2)) * $(262144)) + (((r18_z / $(16384)) % $(16)) * $(16384)) + (((r18_z / $(128)) % $(128)) * $(128)) + (r18_z % $(128))), dtype=tl.bfloat16))
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(524288)])
    tl.store($(vadd_2_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(524288))
    r19_value = (tl.load($(input0_ptr) + ((((r19_z / $(262144)) % $(2)) * $(262144)) + (((r19_z / $(16384)) % $(16)) * $(16384)) + (((r19_z / $(128)) % $(128)) * $(128)) + (r19_z % $(128))), dtype=tl.bfloat16) * tl.load($(vto_copy_3_ptr) + ((((r19_z / $(128)) % $(128)) * $(128)) + (r19_z % $(128))), dtype=tl.bfloat16))
    r19_value = r19_value.to(tl.bfloat16)
    r19_value = tl.broadcast_to(r19_value, [$(524288)])
    tl.store($(vmul_10_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(524288))
    r20_value = (tl.load($(vadd_2_ptr) + ((((r20_z / $(262144)) % $(2)) * $(262144)) + (((r20_z / $(16384)) % $(16)) * $(16384)) + (((r20_z / $(128)) % $(128)) * $(128)) + (r20_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmul_10_ptr) + ((((r20_z / $(262144)) % $(2)) * $(262144)) + (((r20_z / $(16384)) % $(16)) * $(16384)) + (((r20_z / $(128)) % $(128)) * $(128)) + (r20_z % $(128))), dtype=tl.bfloat16))
    r20_value = r20_value.to(tl.bfloat16)
    r20_value = tl.broadcast_to(r20_value, [$(524288)])
    tl.store($(vadd_3_ptr) + r20_z, r20_value)
    z = tl.arange($(524288))
    value = tl.load($(vadd_3_ptr) + ((((z / $(262144)) % $(2)) * $(262144)) + (((z / $(16384)) % $(16)) * $(16384)) + (((z / $(128)) % $(128)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vcos_ptr : RegionName := "vcos"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vsin_ptr : RegionName := "vsin"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vadd_2_ptr : RegionName := "vadd_2"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vreciprocal_ptr : RegionName := "candidate_vreciprocal"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vcos_ptr : RegionName := "candidate_vcos"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vsin_ptr : RegionName := "candidate_vsin"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vcat_3_ptr : RegionName := "candidate_vcat_3"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  triton {
    r0_z = tl.arange($(64))
    r0_value = tl.toReal((tl.cast(r0_z, tl.int64) * tl.full([], $(2), dtype=tl.int64)))
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(64)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(64))
    r1_value = (tl.load($(candidate_vto_copy_ptr) + r1_z, dtype=tl.float32) / 128.0)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(64)])
    tl.store($(candidate_vdiv_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(64))
    r2_value = tl.extra.cuda.libdevice.pow($(10000), tl.load($(candidate_vdiv_ptr) + r2_z, dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(64)])
    tl.store($(candidate_vpow_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(64))
    r3_value = 1.0 / tl.load($(candidate_vpow_1_ptr) + r3_z, dtype=tl.float32)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(64)])
    tl.store($(candidate_vreciprocal_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(64))
    r4_value = (tl.load($(candidate_vreciprocal_ptr) + r4_z, dtype=tl.float32) * 1.0)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(64)])
    tl.store($(candidate_vmul_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(8192))
    r5_value = (((r5_z / $(64)) % $(128)) * tl.load($(candidate_vmul_ptr) + (r5_z % $(64)), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(8192)])
    tl.store($(candidate_vmul_1_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(16384))
    r6_value = tl.where((r6_z % $(128)) < $(64), tl.load($(candidate_vmul_1_ptr) + ((((r6_z / $(128)) % $(128)) * $(64)) + (((r6_z % $(128)) - $(0)) % $(64))), dtype=tl.float32), tl.load($(candidate_vmul_1_ptr) + ((((r6_z / $(128)) % $(128)) * $(64)) + (((r6_z % $(128)) - $(64)) % $(64))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(16384)])
    tl.store($(candidate_vcat_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(16384))
    r7_value = tl.cos(tl.load($(candidate_vcat_ptr) + ((((r7_z / $(128)) % $(128)) * $(128)) + (r7_z % $(128))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(16384)])
    tl.store($(candidate_vcos_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(16384))
    r8_value = (tl.load($(candidate_vcos_ptr) + ((((r8_z / $(128)) % $(128)) * $(128)) + (r8_z % $(128))), dtype=tl.float32) * 1.0)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(16384)])
    tl.store($(candidate_vmul_3_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(16384))
    r9_value = tl.sin(tl.load($(candidate_vcat_ptr) + ((((r9_z / $(128)) % $(128)) * $(128)) + (r9_z % $(128))), dtype=tl.float32))
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(16384)])
    tl.store($(candidate_vsin_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(16384))
    r10_value = (tl.load($(candidate_vsin_ptr) + ((((r10_z / $(128)) % $(128)) * $(128)) + (r10_z % $(128))), dtype=tl.float32) * 1.0)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(16384)])
    tl.store($(candidate_vmul_4_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(16384))
    r11_value = tl.load($(candidate_vmul_3_ptr) + ((((r11_z / $(128)) % $(128)) * $(128)) + (r11_z % $(128))), dtype=tl.float32)
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(16384)])
    tl.store($(candidate_vto_copy_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(16384))
    r12_value = tl.load($(candidate_vmul_4_ptr) + ((((r12_z / $(128)) % $(128)) * $(128)) + (r12_z % $(128))), dtype=tl.float32)
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(16384)])
    tl.store($(candidate_vto_copy_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(524288))
    r13_value = (tl.load($(input0_ptr) + ((((r13_z / $(2048)) % $(2)) * $(262144)) + (((r13_z / $(128)) % $(16)) * $(16384)) + (((r13_z / $(4096)) % $(128)) * $(128)) + (r13_z % $(128))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_3_ptr) + ((((r13_z / $(4096)) % $(128)) * $(128)) + (r13_z % $(128))), dtype=tl.bfloat16))
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(524288)])
    tl.store($(candidate_vmul_7_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(262144))
    r14_value = -tl.load($(candidate_vmul_7_ptr) + ((((r14_z / $(2048)) % $(128)) * $(4096)) + (((r14_z / $(1024)) % $(2)) * $(2048)) + (((r14_z / $(64)) % $(16)) * $(128)) + (r14_z % $(64))), dtype=tl.bfloat16)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(262144)])
    tl.store($(candidate_vneg_1_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(524288))
    r15_value = tl.where((r15_z % $(128)) < $(64), tl.load($(candidate_vmul_7_ptr) + ((((r15_z / $(4096)) % $(128)) * $(4096)) + (((r15_z / $(2048)) % $(2)) * $(2048)) + (((r15_z / $(128)) % $(16)) * $(128)) + ((((r15_z % $(128)) - $(0)) + $(64)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vneg_1_ptr) + ((((r15_z / $(4096)) % $(128)) * $(2048)) + (((r15_z / $(2048)) % $(2)) * $(1024)) + (((r15_z / $(128)) % $(16)) * $(64)) + (((r15_z % $(128)) - $(64)) % $(64))), dtype=tl.bfloat16))
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(524288)])
    tl.store($(candidate_vcat_3_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(524288))
    r16_value = (tl.load($(input0_ptr) + ((((r16_z / $(2048)) % $(2)) * $(262144)) + (((r16_z / $(128)) % $(16)) * $(16384)) + (((r16_z / $(4096)) % $(128)) * $(128)) + (r16_z % $(128))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_2_ptr) + ((((r16_z / $(4096)) % $(128)) * $(128)) + (r16_z % $(128))), dtype=tl.bfloat16))
    r16_value = r16_value.to(tl.bfloat16)
    r16_value = tl.broadcast_to(r16_value, [$(524288)])
    tl.store($(candidate_vmul_8_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(524288))
    r17_value = (tl.load($(candidate_vcat_3_ptr) + ((((r17_z / $(4096)) % $(128)) * $(4096)) + (((r17_z / $(2048)) % $(2)) * $(2048)) + (((r17_z / $(128)) % $(16)) * $(128)) + (r17_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_8_ptr) + ((((r17_z / $(4096)) % $(128)) * $(4096)) + (((r17_z / $(2048)) % $(2)) * $(2048)) + (((r17_z / $(128)) % $(16)) * $(128)) + (r17_z % $(128))), dtype=tl.bfloat16))
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(524288)])
    tl.store($(candidate_vadd_1_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(524288))
    r18_value = tl.where((((r18_z % $(128)) >= $(128)) & ((r18_z % $(128)) < $(128)) & (((r18_z % $(128)) - $(128)) % $(1) == $(0))), tl.load($(input0_ptr) + ((((r18_z / $(2048)) % $(2)) * $(262144)) + (((r18_z / $(128)) % $(16)) * $(16384)) + (((r18_z / $(4096)) % $(128)) * $(128)) + ((((r18_z % $(128)) - $(128)) + $(128)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(524288)])
    tl.store($(candidate_vslice_backward_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(524288))
    r19_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r19_z / $(4096)) % $(128)) * $(4096)) + (((r19_z / $(2048)) % $(2)) * $(2048)) + (((r19_z / $(128)) % $(16)) * $(128)) + (r19_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_1_ptr) + ((((r19_z / $(4096)) % $(128)) * $(4096)) + (((r19_z / $(2048)) % $(2)) * $(2048)) + (((r19_z / $(128)) % $(16)) * $(128)) + (r19_z % $(128))), dtype=tl.bfloat16))
    r19_value = r19_value.to(tl.bfloat16)
    r19_value = tl.broadcast_to(r19_value, [$(524288)])
    tl.store($(candidate_vadd_2_ptr) + r19_z, r19_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vadd_2_ptr) + ((((z / $(128)) % $(128)) * $(4096)) + (((z / $(262144)) % $(2)) * $(2048)) + (((z / $(16384)) % $(16)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_rope_dx
