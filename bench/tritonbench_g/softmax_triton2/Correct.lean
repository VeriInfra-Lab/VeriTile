import bench.tritonbench_g.softmax_triton2.Proofs.ReaderFormat

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.Correct
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 Contiguous
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- The selected FP32 source reads and writes one contiguous masked row.
Storage offsets, row windows and the tail mask are visible on the public IO. -/
def softmaxIO (out inp : RegionName) (ib ob n B : Nat) : MaskedKernelIO₁ :=
  RealProof.softmaxIO out inp ib ob n B

/-- A tile covering all `n` logical columns computes their ordinary softmax.
The complete source succeeds under active access bounds and preserves every
exterior flat-memory cell. Padding is not part of the mathematical sum. -/
specification softmax_kernel_correctness (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) (cover : n ≤ B) :
    Spec.Real (softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (Logical.active cover xs j)) :=
  ReaderFormat.correct out inp ib ob n B hB cover

#specNonCircular reference avoiding [SoftmaxTriton2.softmax_kernel]
#axiomsClean softmax_kernel_correctness
#stmtSurfaceSubset softmax_kernel_correctness ⊆
  [RegionName,Spec.Real,softmaxIO,MaskedKernelIO₁.Implements,MaskedKernelIO₁.B,Logical.active]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton2.Correct
