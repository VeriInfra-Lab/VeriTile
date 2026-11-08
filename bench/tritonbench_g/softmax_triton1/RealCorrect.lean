import bench.tritonbench_g.softmax_triton1.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.Correct
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- The canonical independent row target. -/
def reference := RealIO.reference

/-- Reuse the canonical source, offsets, active windows and masks. -/
def softmaxIO := RealIO.softmaxIO

/-- Successful mathematical execution, access bounds, each active output,
and preservation of every exterior flat-memory cell. -/
specification softmax_kernel_correctness (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) (cover : n ≤ B) :
    Spec.Real (softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (Logical.active cover xs j)) :=
  (Logical.judgment_iff out inp ib ob n B cover).2
    (RealIO.original_logical_correct out inp ib ob n B hB cover)

#specNonCircular reference avoiding [softmax_kernel,reciprocal_kernel]
#axiomsClean softmax_kernel_correctness
#stmtSurfaceSubset softmax_kernel_correctness ⊆
  [RegionName, Spec.Real, softmaxIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B,Logical.active]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.Correct
