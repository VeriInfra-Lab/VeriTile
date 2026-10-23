import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.FP

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelCounterexamples
open VeriTile.Triton

def tiny : Word := FP.Value.ofBits .fp32 0x35800000
def singleton : Tree := .leaf 0
def row : Nat → Word := fun _ => tiny

/-- The historical software-reference defined-cast premise does not follow
even from finite positive FP32 input. This is a model-domain diagnostic,
not a native GPU byte observation or an extra input restriction. -/
theorem tiny_is_finite : tiny.toRat? = some (1 / 1048576) := by
  decide +kernel

theorem half_scale_underflows :
    ((sourceRegisters (softwareOps (fun _ => 0)) singleton row).2).bits.toNat = 0 := by
  decide +kernel

theorem partial_conversion_undefined :
    FP.IntegerConversion.toSigned 8
      (quotient (softwareOps (fun _ => 0)) singleton row 0) = none := by
  decide +kernel

theorem arbitrary_extension_observed :
    (sourceRegisters (softwareOps (fun _ => 0)) singleton row).1 0 ≠
      (sourceRegisters (softwareOps (fun _ => 1)) singleton row).1 0 := by
  decide +kernel

#axiomsClean tiny_is_finite
#axiomsClean half_scale_underflows
#axiomsClean partial_conversion_undefined
#axiomsClean arbitrary_extension_observed
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelCounterexamples
