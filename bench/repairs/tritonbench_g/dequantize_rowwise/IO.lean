import bench.repairs.tritonbench_g.dequantize_rowwise.Kernels
import VeriTile.Triton.Memory.RegionalKernelSpec
import VeriTile.Triton.Memory.LogicalIO

namespace VeriTile.Bench.Repairs.DequantizeRowwise
open VeriTile VeriTile.Triton
noncomputable section

/-- Each invocation reads a masked INT8 row and one unmasked FP32 scale.
The scale channel still has one lane when the output row has zero columns.
Regional IO permits overlapping read-only backing views. -/
abbrev dequantizeIO (p : Parameters) : MixedMaskedKernelIO₂ where
  source := kernel p
  projection := by
    have h : (kernel p).toAlgorithm? = .ok (kernel p).toAlgKernel := rfl
    simp only [ComputeKernel.eraseDType, h, ComputeKernel.toAlgorithm?_fromKernelBody,
      ComputeKernel.toAlgKernel_fromKernelBody]
  in1 := xRegion
  in2 := scaleRegion
  out := outRegion
  type1 := .int
  type2 := .float
  outType := .float
  B1 := p.width
  B2 := 1
  Bout := p.width
  read1 := fun pid j => pid * p.cols + j.val
  read2 := fun pid _ => pid
  write := fun pid j => pid * p.cols + j.val
  mask1 := fun _ j => j.val < p.cols
  mask2 := fun _ _ => True
  outMask := fun _ j => j.val < p.cols

end
end VeriTile.Bench.Repairs.DequantizeRowwise

namespace VeriTile.Bench.Repairs.DequantizeRowwise
open VeriTile.Triton
noncomputable section

structure LogicalInputs (p : Parameters) where
  weights : Fin p.width → Int
  scale : ℝ

def inputEquiv (p : Parameters) :
    ((Fin p.width → Int) × (Fin 1 → ℝ)) ≃ LogicalInputs p where
  toFun := fun xs => ⟨xs.1,xs.2 0⟩
  invFun := fun xs => (xs.weights,fun _ => xs.scale)
  left_inv := by
    rintro ⟨weights,scale⟩
    change (weights, fun _ : Fin 1 => scale 0) = (weights,scale)
    apply congrArg (fun f : Fin 1 → ℝ => (weights,f))
    funext j
    fin_cases j
    rfl
  right_inv := by intro xs; cases xs; rfl

/-- Keep the exact masked row invocation and its unmasked scalar-scale input. -/
def logicalIO (p : Parameters) :
    LogicalIO ((Fin p.width → Int) × (Fin 1 → ℝ)) (Fin p.width → ℝ)
      (LogicalInputs p) (Fin p.width → ℝ)
      (fun f => (dequantizeIO p).Implements (fun weights scale => f (weights,scale))) where
  inputs := inputEquiv p
  outputs := Equiv.refl _

end
end VeriTile.Bench.Repairs.DequantizeRowwise
