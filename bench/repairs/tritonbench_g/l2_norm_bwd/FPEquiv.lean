import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.ProgramPair

namespace VeriTile.Bench.Repairs.L2NormBwd.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- The repaired source and its row-index-commuted candidate execute the full
row grid safely and successfully, agree on every DX cell, and preserve both
complete exterior frames. The candidate only changes the three integer row
products. All FP32 operations, reductions, runtime eps and casts stay intact. -/
specification grid_equiv {A : Type} [Inhabited A] (M : Algebra A) (layout : Strides)
    (rows N block : Nat) (cover : N≤block) (unique : PublicIO.OutputUnique layout rows N)
    (eps : A) :
    original M layout rows N block eps ≡[[]] candidate M layout rows N block eps :=
  ProgramPair.grid_equiv M layout rows N block cover unique eps

#axiomsClean grid_equiv
end
end VeriTile.Bench.Repairs.L2NormBwd.FPEquiv
