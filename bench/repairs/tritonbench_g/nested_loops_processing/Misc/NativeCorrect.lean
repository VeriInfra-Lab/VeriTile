import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.IOImplements

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section

abbrev nestedIO (M : Algebra α) (rows cols : Nat) := IO.nestedIO M Bool.false rows cols
abbrev commuted (M : Algebra α) (rows cols : Nat) := IO.commuted M rows cols

/-- The retained explicit complete-cell formulation underlying the public
two-implementation equivalence. No floating payload arithmetic is performed. -/
theorem kernel_equiv_io {α : Type} [Inhabited α] (M : Algebra α)
    (rows cols : Nat) (positive : 0 < rows*cols) :
    (nestedIO M rows cols).EquivalentOnCells (commuted M rows cols)
      (IO.initialized rows cols) := IOProof.equivalent M rows cols positive

#axiomsClean kernel_equiv_io
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.FPEquiv
