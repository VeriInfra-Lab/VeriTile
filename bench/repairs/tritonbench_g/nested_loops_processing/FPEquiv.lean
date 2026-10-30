import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.Pair

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev original := @PairIO.original
abbrev indexCommuted := @PairIO.indexCommuted
abbrev assumptions := @PairIO.assumptions

/-- Both complete integer programs succeed, have equal typed outputs and
preserve the full exterior frame. The genuine initial multiplication rewrite
changes only indices: no floating numerical assumption is involved. -/
specification kernel_equiv {α : Type} [Inhabited α] (M : Algebra α)
    (rows cols : Nat) (positive : 0 < rows*cols) :
    original M rows cols ≡[assumptions M rows cols] indexCommuted M rows cols :=
  PairProof.equivalent M rows cols positive

#axiomsClean kernel_equiv
#print_fp_assumptions kernel_equiv
#stmtSurfaceSubset kernel_equiv ⊆
  [FP.Structural.Algebra,original,indexCommuted,assumptions,VeriTile.Spec.Equiv,
    ProgramIO.Implementation.programSyntax,ProgramIO.Implementation,
    ProgramIO.InputBindingMode.exactCells,IO.nestedIO,IO.initialized]
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.FPEquiv
