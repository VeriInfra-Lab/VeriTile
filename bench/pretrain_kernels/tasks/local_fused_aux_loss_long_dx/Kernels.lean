import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_aux_loss_long_dx
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
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  triton {
    r0_z = tl.arange($(1))
    r0_value = (tl.load($(input1_ptr) + $(0), dtype=tl.float32) * 0.000000012715657552083334)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(1)])
    tl.store($(vmul_2_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(128))
    r1_value = (tl.load($(vmul_2_ptr) + $(0), dtype=tl.float32) * tl.load($(input0_ptr) + r1_z, dtype=tl.float32))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(128)])
    tl.store($(vmul_3_ptr) + r1_z, r1_value)
    z = tl.arange($(524288))
    value = tl.load($(vmul_3_ptr) + (z % $(128)), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let candidate_vaux_backward_ptr : RegionName := "candidate_vaux_backward"
  triton {
    r0_z = tl.arange($(524288))
    r0_value = ((tl.load($(input1_ptr) + $(0), dtype=tl.float32) * 0.000000012715657552083334) * tl.load($(input0_ptr) + (r0_z % $(128)), dtype=tl.float32))
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(524288)])
    tl.store($(candidate_vaux_backward_ptr) + r0_z, r0_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vaux_backward_ptr) + ((((z / $(128)) % $(4096)) * $(128)) + (z % $(128))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_aux_loss_long_dx
