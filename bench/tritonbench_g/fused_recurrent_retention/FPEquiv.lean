import bench.tritonbench_g.fused_recurrent_retention.Proofs.Pair

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
open VeriTile.Triton FP.Structural Quoted
open scoped VeriTile.Spec
noncomputable section

/-- The historical compare-right candidate on each complete forward/backward
three-axis grid. Only the exact natural key-mask comparison changes orientation;
runtime scale, FP32 arithmetic, reduction plans, optional channels, arbitrary
input tags and strides retain their original scope. -/
specification compare_right_equiv {α : Type} [Inhabited α] (mode : Mode) (a : Arithmetic α)
    (s : Shape) (shape : s.Admitted) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) :
    Pair.original mode a s p incoming finalGradient seed final ≡[[]]
      Pair.compareRight mode a s p incoming finalGradient seed final :=
  ProgramIO.Implementation.of_equivalent (Pair.variant mode a s p incoming finalGradient seed final)
    (Pair.equivalent mode a s shape p incoming finalGradient seed final)

#axiomsClean compare_right_equiv
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
