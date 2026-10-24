import bench.repairs.tritonbench_g.triton_softmax.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.TritonSoftmax.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.LogicalIO
noncomputable section

/-- Unshifted mathematical softmax over exactly the active logical columns. -/
def reference (n B : Nat) (xs : TileIndex [B] → ℝ) (i : TileIndex [B]) : ℝ :=
  Real.exp (xs i) / ∑ j : Fin B, if j.val<n then Real.exp (xs (j,PUnit.unit)) else 0

/-- Shared source, precision, strides, masks and windows are in `IO.lean`. -/
abbrev softmaxIO := IO.rowIO

/-- Every active output is the independent row softmax. The shared IO judgment
also includes successful source execution, actual access safety and exterior frame. -/
specification kernel_real (out inp : RegionName) (ib ob ir ic orow oc n B : Nat)
    (positive : 0<B) (cover : n≤B)
    (injective : ∀ i j : Fin B,i.val<n → j.val<n → i.val*oc=j.val*oc → i=j) :
    Spec.Real (softmaxIO out inp ib ob ir ic orow oc n B ⊨ fun context i =>
      Real.exp (context.2.values i) / ∑ j : Fin n, Real.exp (context.2.active cover j)) := by
  apply (IO.row_iff out inp ib ob ir ic orow oc n B cover).2
  simpa only [LogicalColumns.sum_eq n B cover] using
    RealIOImplements.correct out inp ib ob ir ic orow oc n B positive cover injective

#specNonCircular reference avoiding [softmax_kernel]
#axiomsClean kernel_real
#stmtSurfaceSubset kernel_real ⊆
  [Spec.Real,softmaxIO,Real.exp,LogicalIO.Implements,RegionName,IO.Row,IO.Row.values,IO.Row.active,TileIndex,MaskedTileKernelIO₁.Implements,IO.softmaxIO,MaskedTileKernelIO₁.shape]
end
end VeriTile.Bench.Repairs.TritonSoftmax.Correct
