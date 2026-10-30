import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_muon_param_update
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
  let vto_copy_ptr : RegionName := "vto_copy"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vaddcmul_ptr : RegionName := "vaddcmul"
  triton {
    r0_z = tl.arange($(386199))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(128733)) % $(3)) * $(128733)) + (((r0_z / $(517)) % $(249)) * $(517)) + (r0_z % $(517))), dtype=tl.float32) * 0.1)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(386199)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(386199))
    r1_value = tl.load($(vmul_ptr) + ((((r1_z / $(128733)) % $(3)) * $(128733)) + (((r1_z / $(517)) % $(249)) * $(517)) + (r1_z % $(517))), dtype=tl.float32)
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(386199)])
    tl.store($(vto_copy_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(386199))
    r2_value = (tl.load($(input0_ptr) + ((((r2_z / $(128733)) % $(3)) * $(128733)) + (((r2_z / $(517)) % $(249)) * $(517)) + (r2_z % $(517))), dtype=tl.float32) * 0.9999)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(386199)])
    tl.store($(vmul_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(386199))
    r3_value = tl.load($(vto_copy_ptr) + ((((r3_z / $(128733)) % $(3)) * $(128733)) + (((r3_z / $(517)) % $(249)) * $(517)) + (r3_z % $(517))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(386199)])
    tl.store($(vto_copy_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(386199))
    r4_value = (tl.load($(vmul_1_ptr) + ((((r4_z / $(128733)) % $(3)) * $(128733)) + (((r4_z / $(517)) % $(249)) * $(517)) + (r4_z % $(517))), dtype=tl.float32) + $(1) * -0.00037 * tl.load($(vto_copy_1_ptr) + ((((r4_z / $(128733)) % $(3)) * $(128733)) + (((r4_z / $(517)) % $(249)) * $(517)) + (r4_z % $(517))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(386199)])
    tl.store($(vaddcmul_ptr) + r4_z, r4_value)
    z = tl.arange($(386199))
    value = tl.load($(vaddcmul_ptr) + ((((z / $(128733)) % $(3)) * $(128733)) + (((z / $(517)) % $(249)) * $(517)) + (z % $(517))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(386199)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vfma_ptr : RegionName := "candidate_vfma"
  triton {
    r0_z = tl.arange($(386199))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(128733)) % $(3)) * $(128733)) + (((r0_z / $(517)) % $(249)) * $(517)) + (r0_z % $(517))), dtype=tl.float32) * 0.1)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(386199)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(386199))
    r1_value = tl.load($(candidate_vmul_ptr) + ((((r1_z / $(128733)) % $(3)) * $(128733)) + (((r1_z / $(517)) % $(249)) * $(517)) + (r1_z % $(517))), dtype=tl.float32)
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(386199)])
    tl.store($(candidate_vto_copy_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(386199))
    r2_value = (tl.load($(input0_ptr) + ((((r2_z / $(128733)) % $(3)) * $(128733)) + (((r2_z / $(517)) % $(249)) * $(517)) + (r2_z % $(517))), dtype=tl.float32) * 0.9999)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(386199)])
    tl.store($(candidate_vmul_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(386199))
    r3_value = tl.load($(candidate_vto_copy_ptr) + ((((r3_z / $(128733)) % $(3)) * $(128733)) + (((r3_z / $(517)) % $(249)) * $(517)) + (r3_z % $(517))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(386199)])
    tl.store($(candidate_vto_copy_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(386199))
    r4_value = tl.fma(tl.cast(-0.00037, tl.float32), tl.cast(tl.load($(candidate_vto_copy_1_ptr) + ((((r4_z / $(128733)) % $(3)) * $(128733)) + (((r4_z / $(517)) % $(249)) * $(517)) + (r4_z % $(517))), dtype=tl.float32), tl.float32), tl.cast(tl.load($(candidate_vmul_1_ptr) + ((((r4_z / $(128733)) % $(3)) * $(128733)) + (((r4_z / $(517)) % $(249)) * $(517)) + (r4_z % $(517))), dtype=tl.float32), tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(386199)])
    tl.store($(candidate_vfma_ptr) + r4_z, r4_value)
    z = tl.arange($(386199))
    value = tl.load($(candidate_vfma_ptr) + ((((z / $(128733)) % $(3)) * $(128733)) + (((z / $(517)) % $(249)) * $(517)) + (z % $(517))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(386199)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_muon_param_update
