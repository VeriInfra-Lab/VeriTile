import bench.tritonbench_g.nested_loops_processing.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.PairProof
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Reuse both complete source executions, all output cells, actual access
safety and both exterior frames under the original exact-cell input domain. -/
theorem equivalent {α : Type} [Inhabited α] (M : Algebra α)
    (rows cols : Nat) (positive : 0 < rows*cols) :
    PairIO.original M rows cols ≡[PairIO.assumptions M rows cols]
      PairIO.indexCommuted M rows cols :=
  ProgramIO.Implementation.of_equivalentOnCells (IO.commuted M rows cols)
    (IOProof.equivalent M rows cols positive)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.PairProof
