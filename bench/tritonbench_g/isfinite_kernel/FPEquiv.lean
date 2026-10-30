import bench.tritonbench_g.isfinite_kernel.Proofs.EquivIO
import VeriTile.Meta.StatementAudit

/-! Both actual typed sources have the same observed output cells and exact
exterior frame. Only natural-number index multiplication is commuted. The
classifier remains an arbitrary primitive interpretation, including on NaNs;
there is no numerical rule, finite-only domain or output-injectivity premise.
-/
namespace VeriTile.Bench.Repairs.IsfiniteKernel.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

specification isfinite_index_commuted_equiv {A : Type} [Inhabited A]
    (precision : Precision) (classify : A → Bool) (input output : View) (n B pid : Nat) :
    originalProgram precision classify input output n B pid ≡[[]]
      commutedProgram precision classify input output n B pid :=
  ProgramIO.Implementation.of_equivalent
    (indexCandidate precision classify input output n B pid)
    (IOProof.equivalent precision classify input output n B pid)

#stmtSurfaceSubset isfinite_index_commuted_equiv ⊆
  [Precision, View, originalProgram, commutedProgram, Spec.Equiv,
   ProgramIO.Implementation, ProgramIO.Implementation.programSyntax,
   classificationIO, ProgramIO.Inputs, ProgramIO.InputBindingMode.decoded,
   Spec.RuleEntry, Spec.ProgramSyntax.Statement]
#print_fp_assumptions isfinite_index_commuted_equiv
-- The shared view retains both complete successful source executions.
#axiomsClean isfinite_index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IsfiniteKernel.FPEquiv
