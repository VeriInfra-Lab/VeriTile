import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_rmsnorm_fwd
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
  let vto_copy_ptr : RegionName := "vto_copy"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_ptr : RegionName := "vadd"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
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
    r5_value = (tl.load($(vto_copy_ptr) + ((((r5_z / $(262144)) % $(2)) * $(262144)) + (((r5_z / $(2048)) % $(128)) * $(2048)) + (r5_z % $(2048))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r5_z / $(262144)) % $(2)) * $(128)) + ((r5_z / $(2048)) % $(128))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(524288)])
    tl.store($(vmul_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(524288))
    r6_value = (tl.cast(tl.load($(parameter0_ptr) + (r6_z % $(2048)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_ptr) + ((((r6_z / $(262144)) % $(2)) * $(262144)) + (((r6_z / $(2048)) % $(128)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(524288)])
    tl.store($(vmul_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(524288))
    r7_value = tl.load($(vmul_1_ptr) + ((((r7_z / $(262144)) % $(2)) * $(262144)) + (((r7_z / $(2048)) % $(128)) * $(2048)) + (r7_z % $(2048))), dtype=tl.float32)
    r7_value = r7_value.to(tl.bfloat16)
    r7_value = tl.broadcast_to(r7_value, [$(524288)])
    tl.store($(vto_copy_1_ptr) + r7_z, r7_value)
    z = tl.arange($(524288))
    value = tl.load($(vto_copy_1_ptr) + ((((z / $(262144)) % $(2)) * $(262144)) + (((z / $(2048)) % $(128)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_ptr : RegionName := "vadd"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
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
    r5_value = (tl.load($(candidate_vto_copy_ptr) + ((((r5_z / $(262144)) % $(2)) * $(262144)) + (((r5_z / $(2048)) % $(128)) * $(2048)) + (r5_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r5_z / $(262144)) % $(2)) * $(128)) + ((r5_z / $(2048)) % $(128))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(524288)])
    tl.store($(candidate_vmul_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(524288))
    r6_value = (tl.load($(candidate_vmul_ptr) + ((((r6_z / $(262144)) % $(2)) * $(262144)) + (((r6_z / $(2048)) % $(128)) * $(2048)) + (r6_z % $(2048))), dtype=tl.float32) * tl.cast(tl.load($(parameter0_ptr) + (r6_z % $(2048)), dtype=tl.bfloat16), tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(524288)])
    tl.store($(candidate_vmul_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(524288))
    r7_value = tl.load($(candidate_vmul_1_ptr) + ((((r7_z / $(262144)) % $(2)) * $(262144)) + (((r7_z / $(2048)) % $(128)) * $(2048)) + (r7_z % $(2048))), dtype=tl.float32)
    r7_value = r7_value.to(tl.bfloat16)
    r7_value = tl.broadcast_to(r7_value, [$(524288)])
    tl.store($(candidate_vto_copy_1_ptr) + r7_z, r7_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vto_copy_1_ptr) + ((((z / $(262144)) % $(2)) * $(262144)) + (((z / $(2048)) % $(128)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_rmsnorm_fwd
