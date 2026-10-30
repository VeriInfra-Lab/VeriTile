import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_rmsnorm_dx
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
  let vto_copy_ptr : RegionName := "vto_copy"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_ptr : RegionName := "vadd"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  triton {
    r0_z = tl.arange($(524288))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(262144)) % $(2)) * $(262144)) + (((r0_z / $(2048)) % $(128)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(524288)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(524288))
    r1_value = ((tl.load($(vto_copy_ptr) + ((((r1_z / $(262144)) % $(2)) * $(262144)) + (((r1_z / $(2048)) % $(128)) * $(2048)) + (r1_z % $(2048))), dtype=tl.float32)) * (tl.load($(vto_copy_ptr) + ((((r1_z / $(262144)) % $(2)) * $(262144)) + (((r1_z / $(2048)) % $(128)) * $(2048)) + (r1_z % $(2048))), dtype=tl.float32)))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(524288)])
    tl.store($(vpow_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_lane = tl.arange($(2048))
    r2_value = (tl.sum(tl.load($(vpow_1_ptr) + ((((r2_z[:, None] / $(128)) % $(2)) * $(262144)) + ((r2_z[:, None] % $(128)) * $(2048)) + r2_lane[None, :]), dtype=tl.float32), axis=1) / 2048.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(vmean_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(256))
    r3_value = (tl.load($(vmean_ptr) + ((((r3_z / $(128)) % $(2)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) + 0.000001)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(256)])
    tl.store($(vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(256))
    r4_value = tl.rsqrt(tl.load($(vadd_ptr) + ((((r4_z / $(128)) % $(2)) * $(128)) + (r4_z % $(128))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(256)])
    tl.store($(vrsqrt_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(524288))
    r5_value = tl.load($(input1_ptr) + ((((r5_z / $(262144)) % $(2)) * $(262144)) + (((r5_z / $(2048)) % $(128)) * $(2048)) + (r5_z % $(2048))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(524288)])
    tl.store($(vto_copy_2_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(524288))
    r6_value = (tl.load($(vto_copy_2_ptr) + ((((r6_z / $(262144)) % $(2)) * $(262144)) + (((r6_z / $(2048)) % $(128)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float32) * tl.cast(tl.load($(parameter0_ptr) + (r6_z % $(2048)), dtype=tl.bfloat16), tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(524288)])
    tl.store($(vmul_2_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(524288))
    r7_value = (tl.load($(vmul_2_ptr) + ((((r7_z / $(262144)) % $(2)) * $(262144)) + (((r7_z / $(2048)) % $(128)) * $(2048)) + (r7_z % $(2048))), dtype=tl.float32) * tl.load($(vto_copy_ptr) + ((((r7_z / $(262144)) % $(2)) * $(262144)) + (((r7_z / $(2048)) % $(128)) * $(2048)) + (r7_z % $(2048))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(524288)])
    tl.store($(vmul_3_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(524288))
    r8_value = (tl.load($(vmul_2_ptr) + ((((r8_z / $(262144)) % $(2)) * $(262144)) + (((r8_z / $(2048)) % $(128)) * $(2048)) + (r8_z % $(2048))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r8_z / $(262144)) % $(2)) * $(128)) + ((r8_z / $(2048)) % $(128))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(524288)])
    tl.store($(vmul_4_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(256))
    r9_lane = tl.arange($(2048))
    r9_value = tl.sum(tl.load($(vmul_3_ptr) + ((((r9_z[:, None] / $(128)) % $(2)) * $(262144)) + ((r9_z[:, None] % $(128)) * $(2048)) + r9_lane[None, :]), dtype=tl.float32), axis=1)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(256)])
    tl.store($(vsum_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(256))
    r10_value = ((tl.load($(vrsqrt_ptr) + ((((r10_z / $(128)) % $(2)) * $(128)) + (r10_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r10_z / $(128)) % $(2)) * $(128)) + (r10_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r10_z / $(128)) % $(2)) * $(128)) + (r10_z % $(128))), dtype=tl.float32)))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(256)])
    tl.store($(vpow_2_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(256))
    r11_value = (tl.load($(vsum_1_ptr) + ((((r11_z / $(128)) % $(2)) * $(128)) + (r11_z % $(128))), dtype=tl.float32) * -0.5)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(256)])
    tl.store($(vmul_5_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(256))
    r12_value = (tl.load($(vmul_5_ptr) + ((((r12_z / $(128)) % $(2)) * $(128)) + (r12_z % $(128))), dtype=tl.float32) * tl.load($(vpow_2_ptr) + ((((r12_z / $(128)) % $(2)) * $(128)) + (r12_z % $(128))), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(256)])
    tl.store($(vmul_6_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(524288))
    r13_value = (tl.load($(vmul_6_ptr) + ((((r13_z / $(262144)) % $(2)) * $(128)) + ((r13_z / $(2048)) % $(128))), dtype=tl.float32) / 2048.0)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(524288)])
    tl.store($(vdiv_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(524288))
    r14_value = ((tl.load($(vto_copy_ptr) + ((((r14_z / $(262144)) % $(2)) * $(262144)) + (((r14_z / $(2048)) % $(128)) * $(2048)) + (r14_z % $(2048))), dtype=tl.float32)))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(524288)])
    tl.store($(vpow_3_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(524288))
    r15_value = (tl.load($(vpow_3_ptr) + ((((r15_z / $(262144)) % $(2)) * $(262144)) + (((r15_z / $(2048)) % $(128)) * $(2048)) + (r15_z % $(2048))), dtype=tl.float32) * 2.0)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(524288)])
    tl.store($(vmul_7_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(524288))
    r16_value = (tl.load($(vdiv_ptr) + ((((r16_z / $(262144)) % $(2)) * $(262144)) + (((r16_z / $(2048)) % $(128)) * $(2048)) + (r16_z % $(2048))), dtype=tl.float32) * tl.load($(vmul_7_ptr) + ((((r16_z / $(262144)) % $(2)) * $(262144)) + (((r16_z / $(2048)) % $(128)) * $(2048)) + (r16_z % $(2048))), dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(524288)])
    tl.store($(vmul_8_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(524288))
    r17_value = (tl.load($(vmul_4_ptr) + ((((r17_z / $(262144)) % $(2)) * $(262144)) + (((r17_z / $(2048)) % $(128)) * $(2048)) + (r17_z % $(2048))), dtype=tl.float32) + tl.load($(vmul_8_ptr) + ((((r17_z / $(262144)) % $(2)) * $(262144)) + (((r17_z / $(2048)) % $(128)) * $(2048)) + (r17_z % $(2048))), dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(524288)])
    tl.store($(vadd_1_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(524288))
    r18_value = tl.load($(vadd_1_ptr) + ((((r18_z / $(262144)) % $(2)) * $(262144)) + (((r18_z / $(2048)) % $(128)) * $(2048)) + (r18_z % $(2048))), dtype=tl.float32)
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(524288)])
    tl.store($(vto_copy_3_ptr) + r18_z, r18_value)
    z = tl.arange($(524288))
    value = tl.load($(vto_copy_3_ptr) + ((((z / $(262144)) % $(2)) * $(262144)) + (((z / $(2048)) % $(128)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_ptr : RegionName := "vadd"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vpow_2_ptr : RegionName := "candidate_vpow_2"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vmul_6_ptr : RegionName := "candidate_vmul_6"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  triton {
    r0_z = tl.arange($(524288))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(262144)) % $(2)) * $(262144)) + (((r0_z / $(2048)) % $(128)) * $(2048)) + (r0_z % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(524288)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(524288))
    r1_value = ((tl.load($(candidate_vto_copy_ptr) + ((((r1_z / $(262144)) % $(2)) * $(262144)) + (((r1_z / $(2048)) % $(128)) * $(2048)) + (r1_z % $(2048))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_ptr) + ((((r1_z / $(262144)) % $(2)) * $(262144)) + (((r1_z / $(2048)) % $(128)) * $(2048)) + (r1_z % $(2048))), dtype=tl.float32)))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(524288)])
    tl.store($(candidate_vpow_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_lane = tl.arange($(2048))
    r2_value = (tl.sum(tl.load($(candidate_vpow_1_ptr) + ((((r2_z[:, None] / $(128)) % $(2)) * $(262144)) + ((r2_z[:, None] % $(128)) * $(2048)) + r2_lane[None, :]), dtype=tl.float32), axis=1) / 2048.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(candidate_vmean_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(256))
    r3_value = (tl.load($(candidate_vmean_ptr) + ((((r3_z / $(128)) % $(2)) * $(128)) + (r3_z % $(128))), dtype=tl.float32) + 0.0000009999999974752427)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(256)])
    tl.store($(candidate_vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(256))
    r4_value = tl.rsqrt(tl.load($(candidate_vadd_ptr) + ((((r4_z / $(128)) % $(2)) * $(128)) + (r4_z % $(128))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(256)])
    tl.store($(candidate_vrsqrt_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(524288))
    r5_value = tl.load($(input1_ptr) + ((((r5_z / $(262144)) % $(2)) * $(262144)) + (((r5_z / $(2048)) % $(128)) * $(2048)) + (r5_z % $(2048))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(524288)])
    tl.store($(candidate_vto_copy_2_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(524288))
    r6_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r6_z / $(262144)) % $(2)) * $(262144)) + (((r6_z / $(2048)) % $(128)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float32) * tl.cast(tl.load($(parameter0_ptr) + (r6_z % $(2048)), dtype=tl.bfloat16), tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(524288)])
    tl.store($(candidate_vmul_2_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(524288))
    r7_value = (tl.load($(candidate_vmul_2_ptr) + ((((r7_z / $(262144)) % $(2)) * $(262144)) + (((r7_z / $(2048)) % $(128)) * $(2048)) + (r7_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + ((((r7_z / $(262144)) % $(2)) * $(262144)) + (((r7_z / $(2048)) % $(128)) * $(2048)) + (r7_z % $(2048))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(524288)])
    tl.store($(candidate_vmul_3_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(524288))
    r8_value = (tl.load($(candidate_vmul_2_ptr) + ((((r8_z / $(262144)) % $(2)) * $(262144)) + (((r8_z / $(2048)) % $(128)) * $(2048)) + (r8_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r8_z / $(262144)) % $(2)) * $(128)) + ((r8_z / $(2048)) % $(128))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(524288)])
    tl.store($(candidate_vmul_4_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(256))
    r9_lane = tl.arange($(2048))
    r9_value = tl.sum(tl.load($(candidate_vmul_3_ptr) + ((((r9_z[:, None] / $(128)) % $(2)) * $(262144)) + ((r9_z[:, None] % $(128)) * $(2048)) + r9_lane[None, :]), dtype=tl.float32), axis=1)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(256)])
    tl.store($(candidate_vsum_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(256))
    r10_value = ((tl.load($(candidate_vrsqrt_ptr) + ((((r10_z / $(128)) % $(2)) * $(128)) + (r10_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r10_z / $(128)) % $(2)) * $(128)) + (r10_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r10_z / $(128)) % $(2)) * $(128)) + (r10_z % $(128))), dtype=tl.float32)))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(256)])
    tl.store($(candidate_vpow_2_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(256))
    r11_value = (tl.load($(candidate_vsum_1_ptr) + ((((r11_z / $(128)) % $(2)) * $(128)) + (r11_z % $(128))), dtype=tl.float32) * -0.5)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(256)])
    tl.store($(candidate_vmul_5_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(256))
    r12_value = (tl.load($(candidate_vmul_5_ptr) + ((((r12_z / $(128)) % $(2)) * $(128)) + (r12_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vpow_2_ptr) + ((((r12_z / $(128)) % $(2)) * $(128)) + (r12_z % $(128))), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(256)])
    tl.store($(candidate_vmul_6_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(524288))
    r13_value = (tl.load($(candidate_vmul_6_ptr) + ((((r13_z / $(262144)) % $(2)) * $(128)) + ((r13_z / $(2048)) % $(128))), dtype=tl.float32) / 2048.0)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(524288)])
    tl.store($(candidate_vdiv_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(524288))
    r14_value = ((tl.load($(candidate_vto_copy_ptr) + ((((r14_z / $(262144)) % $(2)) * $(262144)) + (((r14_z / $(2048)) % $(128)) * $(2048)) + (r14_z % $(2048))), dtype=tl.float32)))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(524288)])
    tl.store($(candidate_vpow_3_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(524288))
    r15_value = (tl.load($(candidate_vpow_3_ptr) + ((((r15_z / $(262144)) % $(2)) * $(262144)) + (((r15_z / $(2048)) % $(128)) * $(2048)) + (r15_z % $(2048))), dtype=tl.float32) * 2.0)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(524288)])
    tl.store($(candidate_vmul_7_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(524288))
    r16_value = (tl.load($(candidate_vdiv_ptr) + ((((r16_z / $(262144)) % $(2)) * $(262144)) + (((r16_z / $(2048)) % $(128)) * $(2048)) + (r16_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vmul_7_ptr) + ((((r16_z / $(262144)) % $(2)) * $(262144)) + (((r16_z / $(2048)) % $(128)) * $(2048)) + (r16_z % $(2048))), dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(524288)])
    tl.store($(candidate_vmul_8_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(524288))
    r17_value = (tl.load($(candidate_vmul_4_ptr) + ((((r17_z / $(262144)) % $(2)) * $(262144)) + (((r17_z / $(2048)) % $(128)) * $(2048)) + (r17_z % $(2048))), dtype=tl.float32) + tl.load($(candidate_vmul_8_ptr) + ((((r17_z / $(262144)) % $(2)) * $(262144)) + (((r17_z / $(2048)) % $(128)) * $(2048)) + (r17_z % $(2048))), dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(524288)])
    tl.store($(candidate_vadd_1_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(524288))
    r18_value = tl.load($(candidate_vadd_1_ptr) + ((((r18_z / $(262144)) % $(2)) * $(262144)) + (((r18_z / $(2048)) % $(128)) * $(2048)) + (r18_z % $(2048))), dtype=tl.float32)
    r18_value = r18_value.to(tl.bfloat16)
    r18_value = tl.broadcast_to(r18_value, [$(524288)])
    tl.store($(candidate_vto_copy_3_ptr) + r18_z, r18_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vto_copy_3_ptr) + ((((z / $(262144)) % $(2)) * $(262144)) + (((z / $(2048)) % $(128)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_rmsnorm_dx
