import bench.tritonbench_g.quantize_global.Proofs.IOCorrect

namespace QuantizeGlobalRepair.PairProof
open VeriTile VeriTile.Triton FP.Structural QuantizeIO
open scoped VeriTile.Spec
noncomputable section

theorem equivalent {α : Type} [Inhabited α]
    (rest : Algebra α) (a : Scalar α) (p : Parameters) (sep : p.Separated)
    (left right : Nat → List Config)
    (leftNonempty : ∀ k, k < segmentCount p.n → left k ≠ [])
    (rightNonempty : ∀ k, k < segmentCount p.n → right k ≠ []) :
    original rest a p left ≡[[]] candidate rest a p left right :=
  IntegerProgramIO.Implementation.equiv_of_equivalent (variant rest a p left right)
    (IOCorrect.equivalent rest a p sep left right leftNonempty rightNonempty)

#axiomsClean equivalent
end
end QuantizeGlobalRepair.PairProof
