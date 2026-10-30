import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.Pair

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- The actual selected and address-commuted sources are safe and terminate,
with equal complete output cells and both exterior frames. This exact integer
rewrite needs no floating numerical assumption or destination injectivity. -/
specification source_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (p : Source.Parameters) (ids : Nat → Nat) :
    original M p ids ≡[[]] candidate M p ids :=
  PairProof.equivalent M p ids

#print_fp_assumptions source_equivalence
#axiomsClean source_equivalence
#stmtSurfaceSubset source_equivalence ⊆
  [FP.Structural.Algebra,Source.Parameters,original,candidate,VeriTile.Spec.Equiv,
    ProgramIO.Implementation,ProgramIO.InputBindingMode.decoded,
    ProgramIO.Implementation.programSyntax,originalIO,ProgramIO.Inputs,
    VeriTile.Spec.RuleEntry,VeriTile.Spec.ProgramSyntax.Statement]
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.FPEquiv
