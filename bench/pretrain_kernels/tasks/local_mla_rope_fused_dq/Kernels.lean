import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_rope_fused_dq
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
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vbmm_ptr : RegionName := "vbmm"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vcos_1_ptr : RegionName := "vcos_1"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vsin_1_ptr : RegionName := "vsin_1"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vneg_ptr : RegionName := "vneg"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_8_ptr : RegionName := "vcat_8"
  triton {
    r0_z = tl.arange($(4))
    r0_value = tl.toReal((tl.cast(r0_z, tl.int64) * tl.full([], $(2), dtype=tl.int64)))
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(4)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(4))
    r1_value = (tl.load($(vto_copy_ptr) + r1_z, dtype=tl.float32) / 8.0)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(4)])
    tl.store($(vdiv_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(4))
    r2_value = tl.extra.cuda.libdevice.pow($(10000), tl.load($(vdiv_ptr) + r2_z, dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(4)])
    tl.store($(vpow_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(4))
    r3_value = 1.0 / tl.load($(vpow_1_ptr) + r3_z, dtype=tl.float32)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(4)])
    tl.store($(vreciprocal_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(4))
    r4_value = (tl.load($(vreciprocal_ptr) + r4_z, dtype=tl.float32) * 1.0)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(4)])
    tl.store($(vmul_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(128))
    r5_value = tl.toReal(tl.cast(r5_z, tl.int64))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(128)])
    tl.store($(vto_copy_4_ptr) + r5_z, r5_value)
    r6_row = tl.arange($(4))
    r6_col = tl.arange($(128))
    r6_lane = tl.arange($(1))
    for r6_group in range($(1)) {
      r6_left = tl.load($(vmul_ptr) + ((r6_row[:, None] + r6_lane[None, :]) % $(4)), dtype=tl.float32)
      r6_right = tl.load($(vto_copy_4_ptr) + (((r6_lane[:, None] * $(128)) + r6_col[None, :]) % $(128)), dtype=tl.float32)
      r6_value = tl.dot(r6_left, r6_right)
      r6_value = r6_value.to(tl.float32)
      tl.store($(vbmm_ptr) + ((r6_group * $(512)) + (r6_row[:, None] * $(128)) + r6_col[None, :]), r6_value)
    }
    r7_z = tl.arange($(1024))
    r7_value = tl.where((r7_z % $(8)) < $(4), tl.load($(vbmm_ptr) + (((((r7_z % $(8)) - $(0)) % $(4)) * $(128)) + ((r7_z / $(8)) % $(128))), dtype=tl.float32), tl.load($(vbmm_ptr) + (((((r7_z % $(8)) - $(4)) % $(4)) * $(128)) + ((r7_z / $(8)) % $(128))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1024)])
    tl.store($(vcat_2_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(1024))
    r8_value = tl.cos(tl.load($(vcat_2_ptr) + ((((r8_z / $(8)) % $(128)) * $(8)) + (r8_z % $(8))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(1024)])
    tl.store($(vcos_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(1024))
    r9_value = (tl.load($(vcos_1_ptr) + ((((r9_z / $(8)) % $(128)) * $(8)) + (r9_z % $(8))), dtype=tl.float32) * 1.25)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(1024)])
    tl.store($(vmul_5_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(1024))
    r10_value = tl.sin(tl.load($(vcat_2_ptr) + ((((r10_z / $(8)) % $(128)) * $(8)) + (r10_z % $(8))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1024)])
    tl.store($(vsin_1_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1024))
    r11_value = (tl.load($(vsin_1_ptr) + ((((r11_z / $(8)) % $(128)) * $(8)) + (r11_z % $(8))), dtype=tl.float32) * 1.25)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(1024)])
    tl.store($(vmul_6_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(1024))
    r12_value = tl.load($(vmul_5_ptr) + ((((r12_z / $(8)) % $(128)) * $(8)) + (r12_z % $(8))), dtype=tl.float32)
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(1024)])
    tl.store($(vto_copy_5_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(1024))
    r13_value = tl.load($(vmul_6_ptr) + ((((r13_z / $(8)) % $(128)) * $(8)) + (r13_z % $(8))), dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(1024)])
    tl.store($(vto_copy_6_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(12288))
    r14_value = (tl.load($(input0_ptr) + ((((r14_z / $(6144)) % $(2)) * $(98304)) + (((r14_z / $(512)) % $(12)) * $(8192)) + (((r14_z / $(4)) % $(128)) * $(64)) + (r14_z % $(4)) + $(20)), dtype=tl.bfloat16) * tl.load($(vto_copy_6_ptr) + ((((r14_z / $(4)) % $(128)) * $(8)) + (r14_z % $(4))), dtype=tl.bfloat16))
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(12288)])
    tl.store($(vmul_15_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(12288))
    r15_value = (tl.load($(input0_ptr) + ((((r15_z / $(6144)) % $(2)) * $(98304)) + (((r15_z / $(512)) % $(12)) * $(8192)) + (((r15_z / $(4)) % $(128)) * $(64)) + (r15_z % $(4)) + $(20)), dtype=tl.bfloat16) * tl.load($(vto_copy_5_ptr) + ((((r15_z / $(4)) % $(128)) * $(8)) + (r15_z % $(4))), dtype=tl.bfloat16))
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(12288)])
    tl.store($(vmul_16_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(12288))
    r16_value = -tl.load($(input0_ptr) + ((((r16_z / $(6144)) % $(2)) * $(98304)) + (((r16_z / $(512)) % $(12)) * $(8192)) + (((r16_z / $(4)) % $(128)) * $(64)) + (r16_z % $(4)) + $(16)), dtype=tl.bfloat16)
    r16_value = r16_value.to(tl.bfloat16)
    r16_value = tl.broadcast_to(r16_value, [$(12288)])
    tl.store($(vneg_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(12288))
    r17_value = (tl.load($(vneg_ptr) + ((((r17_z / $(6144)) % $(2)) * $(6144)) + (((r17_z / $(512)) % $(12)) * $(512)) + (((r17_z / $(4)) % $(128)) * $(4)) + (r17_z % $(4))), dtype=tl.bfloat16) * tl.load($(vto_copy_6_ptr) + ((((r17_z / $(4)) % $(128)) * $(8)) + (r17_z % $(4))), dtype=tl.bfloat16))
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(12288)])
    tl.store($(vmul_17_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(12288))
    r18_value = (tl.load($(vmul_16_ptr) + ((((r18_z / $(6144)) % $(2)) * $(6144)) + (((r18_z / $(512)) % $(12)) * $(512)) + (((r18_z / $(4)) % $(128)) * $(4)) + (r18_z % $(4))), dtype=tl.bfloat16) + tl.load($(vmul_17_ptr) + ((((r18_z / $(6144)) % $(2)) * $(6144)) + (((r18_z / $(512)) % $(12)) * $(512)) + (((r18_z / $(4)) % $(128)) * $(4)) + (r18_z % $(4))), dtype=tl.bfloat16))
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(12288)])
    tl.store($(vadd_2_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(12288))
    r19_value = (tl.load($(input0_ptr) + ((((r19_z / $(6144)) % $(2)) * $(98304)) + (((r19_z / $(512)) % $(12)) * $(8192)) + (((r19_z / $(4)) % $(128)) * $(64)) + (r19_z % $(4)) + $(16)), dtype=tl.bfloat16) * tl.load($(vto_copy_5_ptr) + ((((r19_z / $(4)) % $(128)) * $(8)) + (r19_z % $(4))), dtype=tl.bfloat16))
    r19_value = r19_value.to(tl.bfloat16)
    r19_value = tl.broadcast_to(r19_value, [$(12288)])
    tl.store($(vmul_18_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(12288))
    r20_value = (tl.load($(vmul_15_ptr) + ((((r20_z / $(6144)) % $(2)) * $(6144)) + (((r20_z / $(512)) % $(12)) * $(512)) + (((r20_z / $(4)) % $(128)) * $(4)) + (r20_z % $(4))), dtype=tl.bfloat16) + tl.load($(vmul_18_ptr) + ((((r20_z / $(6144)) % $(2)) * $(6144)) + (((r20_z / $(512)) % $(12)) * $(512)) + (((r20_z / $(4)) % $(128)) * $(4)) + (r20_z % $(4))), dtype=tl.bfloat16))
    r20_value = r20_value.to(tl.bfloat16)
    r20_value = tl.broadcast_to(r20_value, [$(12288)])
    tl.store($(vadd_3_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(24576))
    r21_value = tl.where((((r21_z % $(8)) >= $(1)) & ((r21_z % $(8)) < $(8)) & (((r21_z % $(8)) - $(1)) % $(2) == $(0))), tl.load($(vadd_2_ptr) + ((((r21_z / $(12288)) % $(2)) * $(6144)) + (((r21_z / $(1024)) % $(12)) * $(512)) + (((r21_z / $(8)) % $(128)) * $(4)) + ((((r21_z % $(8)) - $(1)) / $(2)) % $(4))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r21_value = r21_value.to(tl.bfloat16)
    r21_value = tl.broadcast_to(r21_value, [$(24576)])
    tl.store($(vslice_backward_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(24576))
    r22_value = tl.where((((r22_z % $(8)) >= $(0)) & ((r22_z % $(8)) < $(8)) & (((r22_z % $(8)) - $(0)) % $(2) == $(0))), tl.load($(vadd_3_ptr) + ((((r22_z / $(12288)) % $(2)) * $(6144)) + (((r22_z / $(1024)) % $(12)) * $(512)) + (((r22_z / $(8)) % $(128)) * $(4)) + ((((r22_z % $(8)) - $(0)) / $(2)) % $(4))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r22_value = r22_value.to(tl.bfloat16)
    r22_value = tl.broadcast_to(r22_value, [$(24576)])
    tl.store($(vslice_backward_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(24576))
    r23_value = (tl.load($(vslice_backward_ptr) + ((((r23_z / $(12288)) % $(2)) * $(12288)) + (((r23_z / $(1024)) % $(12)) * $(1024)) + (((r23_z / $(8)) % $(128)) * $(8)) + (r23_z % $(8))), dtype=tl.bfloat16) + tl.load($(vslice_backward_1_ptr) + ((((r23_z / $(12288)) % $(2)) * $(12288)) + (((r23_z / $(1024)) % $(12)) * $(1024)) + (((r23_z / $(8)) % $(128)) * $(8)) + (r23_z % $(8))), dtype=tl.bfloat16))
    r23_value = r23_value.to(tl.bfloat16)
    r23_value = tl.broadcast_to(r23_value, [$(24576)])
    tl.store($(vadd_4_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(73728))
    r24_value = tl.where((r24_z % $(24)) < $(16), tl.load($(input0_ptr) + ((((r24_z / $(36864)) % $(2)) * $(98304)) + (((r24_z / $(3072)) % $(12)) * $(8192)) + (((r24_z / $(24)) % $(128)) * $(64)) + (((r24_z % $(24)) - $(0)) % $(64))), dtype=tl.bfloat16), tl.load($(vadd_4_ptr) + ((((r24_z / $(36864)) % $(2)) * $(12288)) + (((r24_z / $(3072)) % $(12)) * $(1024)) + (((r24_z / $(24)) % $(128)) * $(8)) + (((r24_z % $(24)) - $(16)) % $(8))), dtype=tl.bfloat16))
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(73728)])
    tl.store($(vcat_8_ptr) + r24_z, r24_value)
    z = tl.arange($(73728))
    value = tl.load($(vcat_8_ptr) + ((((z / $(36864)) % $(2)) * $(36864)) + (((z / $(3072)) % $(12)) * $(3072)) + (((z / $(24)) % $(128)) * $(24)) + (z % $(24))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(73728)])
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
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vneg_ptr : RegionName := "vneg"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_4_ptr : RegionName := "vadd_4"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vreciprocal_ptr : RegionName := "candidate_vreciprocal"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vcos_ptr : RegionName := "candidate_vcos"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vcat_1_ptr : RegionName := "candidate_vcat_1"
  let candidate_vsin_ptr : RegionName := "candidate_vsin"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  triton {
    r0_z = tl.arange($(4))
    r0_value = tl.toReal((tl.cast(r0_z, tl.int64) * tl.full([], $(2), dtype=tl.int64)))
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(4)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(4))
    r1_value = (tl.load($(candidate_vto_copy_ptr) + r1_z, dtype=tl.float32) / 8.0)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(4)])
    tl.store($(candidate_vdiv_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(4))
    r2_value = tl.extra.cuda.libdevice.pow($(10000), tl.load($(candidate_vdiv_ptr) + r2_z, dtype=tl.float32))
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(4)])
    tl.store($(candidate_vpow_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(4))
    r3_value = 1.0 / tl.load($(candidate_vpow_1_ptr) + r3_z, dtype=tl.float32)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(4)])
    tl.store($(candidate_vreciprocal_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(4))
    r4_value = (tl.load($(candidate_vreciprocal_ptr) + r4_z, dtype=tl.float32) * 1.0)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(4)])
    tl.store($(candidate_vmul_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = (((r5_z / $(4)) % $(128)) * tl.load($(candidate_vmul_ptr) + (r5_z % $(4)), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(candidate_vmul_1_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(1024))
    r6_value = tl.where((r6_z % $(8)) < $(4), tl.load($(candidate_vmul_1_ptr) + ((((r6_z / $(8)) % $(128)) * $(4)) + (((r6_z % $(8)) - $(0)) % $(4))), dtype=tl.float32), tl.load($(candidate_vmul_1_ptr) + ((((r6_z / $(8)) % $(128)) * $(4)) + (((r6_z % $(8)) - $(4)) % $(4))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(1024)])
    tl.store($(candidate_vcat_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1024))
    r7_value = tl.cos(tl.load($(candidate_vcat_ptr) + ((((r7_z / $(8)) % $(128)) * $(8)) + (r7_z % $(8))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1024)])
    tl.store($(candidate_vcos_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(1024))
    r8_value = (tl.load($(candidate_vcos_ptr) + ((((r8_z / $(8)) % $(128)) * $(8)) + (r8_z % $(8))), dtype=tl.float32) * 1.25)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(1024)])
    tl.store($(candidate_vmul_2_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(1024))
    r9_value = tl.load($(candidate_vmul_2_ptr) + ((((r9_z / $(8)) % $(128)) * $(8)) + (r9_z % $(8))), dtype=tl.float32)
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(1024)])
    tl.store($(candidate_vto_copy_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(1024))
    r10_value = tl.where((r10_z % $(8)) < $(4), tl.load($(candidate_vmul_1_ptr) + ((((r10_z / $(8)) % $(128)) * $(4)) + (((r10_z % $(8)) - $(0)) % $(4))), dtype=tl.float32), tl.load($(candidate_vmul_1_ptr) + ((((r10_z / $(8)) % $(128)) * $(4)) + (((r10_z % $(8)) - $(4)) % $(4))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1024)])
    tl.store($(candidate_vcat_1_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(1024))
    r11_value = tl.sin(tl.load($(candidate_vcat_1_ptr) + ((((r11_z / $(8)) % $(128)) * $(8)) + (r11_z % $(8))), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(1024)])
    tl.store($(candidate_vsin_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(1024))
    r12_value = (tl.load($(candidate_vsin_ptr) + ((((r12_z / $(8)) % $(128)) * $(8)) + (r12_z % $(8))), dtype=tl.float32) * 1.25)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(1024)])
    tl.store($(candidate_vmul_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(1024))
    r13_value = tl.load($(candidate_vmul_3_ptr) + ((((r13_z / $(8)) % $(128)) * $(8)) + (r13_z % $(8))), dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(1024)])
    tl.store($(candidate_vto_copy_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(12288))
    r14_value = (tl.load($(input0_ptr) + ((((r14_z / $(48)) % $(2)) * $(98304)) + (((r14_z / $(4)) % $(12)) * $(8192)) + (((r14_z / $(96)) % $(128)) * $(64)) + (r14_z % $(4)) + $(20)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_2_ptr) + ((((r14_z / $(96)) % $(128)) * $(8)) + (r14_z % $(4)) + $(4)), dtype=tl.bfloat16))
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(12288)])
    tl.store($(candidate_vmul_13_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(12288))
    r15_value = (tl.load($(input0_ptr) + ((((r15_z / $(48)) % $(2)) * $(98304)) + (((r15_z / $(4)) % $(12)) * $(8192)) + (((r15_z / $(96)) % $(128)) * $(64)) + (r15_z % $(4)) + $(20)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_1_ptr) + ((((r15_z / $(96)) % $(128)) * $(8)) + (r15_z % $(4)) + $(4)), dtype=tl.bfloat16))
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(12288)])
    tl.store($(candidate_vmul_14_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(12288))
    r16_value = -tl.load($(input0_ptr) + ((((r16_z / $(48)) % $(2)) * $(98304)) + (((r16_z / $(4)) % $(12)) * $(8192)) + (((r16_z / $(96)) % $(128)) * $(64)) + (r16_z % $(4)) + $(16)), dtype=tl.bfloat16)
    r16_value = r16_value.to(tl.bfloat16)
    r16_value = tl.broadcast_to(r16_value, [$(12288)])
    tl.store($(candidate_vneg_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(12288))
    r17_value = (tl.load($(candidate_vneg_ptr) + ((((r17_z / $(96)) % $(128)) * $(96)) + (((r17_z / $(48)) % $(2)) * $(48)) + (((r17_z / $(4)) % $(12)) * $(4)) + (r17_z % $(4))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_2_ptr) + ((((r17_z / $(96)) % $(128)) * $(8)) + (r17_z % $(4))), dtype=tl.bfloat16))
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(12288)])
    tl.store($(candidate_vmul_15_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(12288))
    r18_value = (tl.load($(candidate_vmul_14_ptr) + ((((r18_z / $(96)) % $(128)) * $(96)) + (((r18_z / $(48)) % $(2)) * $(48)) + (((r18_z / $(4)) % $(12)) * $(4)) + (r18_z % $(4))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_15_ptr) + ((((r18_z / $(96)) % $(128)) * $(96)) + (((r18_z / $(48)) % $(2)) * $(48)) + (((r18_z / $(4)) % $(12)) * $(4)) + (r18_z % $(4))), dtype=tl.bfloat16))
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(12288)])
    tl.store($(candidate_vadd_2_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(12288))
    r19_value = (tl.load($(input0_ptr) + ((((r19_z / $(48)) % $(2)) * $(98304)) + (((r19_z / $(4)) % $(12)) * $(8192)) + (((r19_z / $(96)) % $(128)) * $(64)) + (r19_z % $(4)) + $(16)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_1_ptr) + ((((r19_z / $(96)) % $(128)) * $(8)) + (r19_z % $(4))), dtype=tl.bfloat16))
    r19_value = r19_value.to(tl.bfloat16)
    r19_value = tl.broadcast_to(r19_value, [$(12288)])
    tl.store($(candidate_vmul_16_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(12288))
    r20_value = (tl.load($(candidate_vmul_13_ptr) + ((((r20_z / $(96)) % $(128)) * $(96)) + (((r20_z / $(48)) % $(2)) * $(48)) + (((r20_z / $(4)) % $(12)) * $(4)) + (r20_z % $(4))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_16_ptr) + ((((r20_z / $(96)) % $(128)) * $(96)) + (((r20_z / $(48)) % $(2)) * $(48)) + (((r20_z / $(4)) % $(12)) * $(4)) + (r20_z % $(4))), dtype=tl.bfloat16))
    r20_value = r20_value.to(tl.bfloat16)
    r20_value = tl.broadcast_to(r20_value, [$(12288)])
    tl.store($(candidate_vadd_3_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(24576))
    r21_value = tl.where((((r21_z % $(8)) >= $(1)) & ((r21_z % $(8)) < $(8)) & (((r21_z % $(8)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_2_ptr) + ((((r21_z / $(192)) % $(128)) * $(96)) + (((r21_z / $(96)) % $(2)) * $(48)) + (((r21_z / $(8)) % $(12)) * $(4)) + ((((r21_z % $(8)) - $(1)) / $(2)) % $(4))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r21_value = r21_value.to(tl.bfloat16)
    r21_value = tl.broadcast_to(r21_value, [$(24576)])
    tl.store($(candidate_vslice_backward_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(24576))
    r22_value = tl.where((((r22_z % $(8)) >= $(0)) & ((r22_z % $(8)) < $(8)) & (((r22_z % $(8)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_3_ptr) + ((((r22_z / $(192)) % $(128)) * $(96)) + (((r22_z / $(96)) % $(2)) * $(48)) + (((r22_z / $(8)) % $(12)) * $(4)) + ((((r22_z % $(8)) - $(0)) / $(2)) % $(4))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r22_value = r22_value.to(tl.bfloat16)
    r22_value = tl.broadcast_to(r22_value, [$(24576)])
    tl.store($(candidate_vslice_backward_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(24576))
    r23_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r23_z / $(192)) % $(128)) * $(192)) + (((r23_z / $(96)) % $(2)) * $(96)) + (((r23_z / $(8)) % $(12)) * $(8)) + (r23_z % $(8))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_1_ptr) + ((((r23_z / $(192)) % $(128)) * $(192)) + (((r23_z / $(96)) % $(2)) * $(96)) + (((r23_z / $(8)) % $(12)) * $(8)) + (r23_z % $(8))), dtype=tl.bfloat16))
    r23_value = r23_value.to(tl.bfloat16)
    r23_value = tl.broadcast_to(r23_value, [$(24576)])
    tl.store($(candidate_vadd_4_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(73728))
    r24_value = tl.where((((r24_z % $(24)) >= $(16)) & ((r24_z % $(24)) < $(24)) & (((r24_z % $(24)) - $(16)) % $(1) == $(0))), tl.load($(candidate_vadd_4_ptr) + ((((r24_z / $(576)) % $(128)) * $(192)) + (((r24_z / $(288)) % $(2)) * $(96)) + (((r24_z / $(24)) % $(12)) * $(8)) + (((r24_z % $(24)) - $(16)) % $(8))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(73728)])
    tl.store($(candidate_vslice_backward_2_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(73728))
    r25_value = tl.where((((r25_z % $(24)) >= $(0)) & ((r25_z % $(24)) < $(16)) & (((r25_z % $(24)) - $(0)) % $(1) == $(0))), tl.load($(input0_ptr) + ((((r25_z / $(288)) % $(2)) * $(98304)) + (((r25_z / $(24)) % $(12)) * $(8192)) + (((r25_z / $(576)) % $(128)) * $(64)) + (((r25_z % $(24)) - $(0)) % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r25_value = r25_value.to(tl.bfloat16)
    r25_value = tl.broadcast_to(r25_value, [$(73728)])
    tl.store($(candidate_vslice_backward_3_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(73728))
    r26_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r26_z / $(576)) % $(128)) * $(576)) + (((r26_z / $(288)) % $(2)) * $(288)) + (((r26_z / $(24)) % $(12)) * $(24)) + (r26_z % $(24))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_3_ptr) + ((((r26_z / $(576)) % $(128)) * $(576)) + (((r26_z / $(288)) % $(2)) * $(288)) + (((r26_z / $(24)) % $(12)) * $(24)) + (r26_z % $(24))), dtype=tl.bfloat16))
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(73728)])
    tl.store($(candidate_vadd_5_ptr) + r26_z, r26_value)
    z = tl.arange($(73728))
    value = tl.load($(candidate_vadd_5_ptr) + ((((z / $(24)) % $(128)) * $(576)) + (((z / $(36864)) % $(2)) * $(288)) + (((z / $(3072)) % $(12)) * $(24)) + (z % $(24))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(73728)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_rope_fused_dq
