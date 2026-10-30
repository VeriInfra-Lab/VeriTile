import bench.repairs.tritonbench_g.softmax_triton2.Proofs.Real
import bench.repairs.tritonbench_g.softmax_triton2.IO

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.ReaderFormat
open VeriTile.Triton
open scoped BigOperators VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- Full old/new judgment equivalence retains the B-wide masked row. -/
theorem statement_iff (out inp : RegionName) (ib ob n B : Nat) (cover : n ≤ B) :
    (RealProof.softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (Logical.active cover xs j)) ↔
    (RealProof.softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (xs (Fin.castLE cover j))) := Iff.rfl

theorem correct (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) (cover : n ≤ B) :
    RealProof.softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (Logical.active cover xs j) :=
  (statement_iff out inp ib ob n B cover).mpr
    (RealProof.logical_io_correct out inp ib ob n B hB cover)

end
end VeriTile.Bench.Repairs.SoftmaxTriton2.ReaderFormat
