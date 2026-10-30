import bench.repairs.tritonbench_g.softmax_triton3.Proofs.LogicalIO
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Correct
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding blockSize numWarps
open scoped VeriTile.Triton.MaskedKernelIO₁ VeriTile.Triton.MaskedKernelIO₂
noncomputable section

/-- Independent last-axis softmax of input plus the optional additive mask. -/
def reference : Formula ℝ := KernelRealResult.reference

/-- Independent mathematical formula over the active logical row columns. -/
def rowFormula (n B : Nat) (xs : Fin B → ℝ) (j : Fin B) : ℝ :=
  Real.exp (xs j) / ∑ k : Fin B, if k.val<n then Real.exp (xs k) else 0

/-- Public IO is in IO.lean: complete FP32 source, active row windows, safe
accesses, successful execution and the frame outside the output window. -/
specification softmax_kernel_correct (out inp : RegionName) (n B : Nat)
    (positive : 0<B) (cover : n≤B) :
    Spec.Real (softmaxIO out inp n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (Logical.active cover xs j)) :=
  Logical.unmasked out inp n B positive cover

/-- The optional mask supplies a second input row to the same independent
mathematical softmax formula. -/
specification softmax_masked_kernel_correct (out inp mask : RegionName) (n B : Nat)
    (positive : 0<B) (cover : n≤B) :
    Spec.Real (softmaxMaskedIO out inp mask n B ⊨ fun xs ms i =>
      Real.exp (xs i + ms i) /
        ∑ j : Fin n, Real.exp (Logical.active cover xs j + Logical.active cover ms j)) :=
  Logical.masked out inp mask n B positive cover

#stmtSurfaceSubset softmax_kernel_correct ⊆
  [Spec.Real,RegionName,softmaxIO,MaskedKernelIO₁.Implements,MaskedKernelIO₁.B,Logical.active]
#stmtSurfaceSubset softmax_masked_kernel_correct ⊆
  [Spec.Real,RegionName,softmaxMaskedIO,MaskedKernelIO₂.Implements,MaskedKernelIO₂.B,Logical.active]
#specNonCircular rowFormula avoiding [softmax_kernel,RealModel.evaluate]
#specNonCircular reference avoiding [softmax_kernel,RealModel.evaluate]
#axiomsClean softmax_kernel_correct
#axiomsClean softmax_masked_kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Correct
