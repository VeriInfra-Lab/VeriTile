import bench.tritonbench_g.l2_norm_triton1.PairIO
import bench.tritonbench_g.l2_norm_triton1.Proofs.GridIO

/-! Reader entry: full-grid equivalence of the actual selected sources.
IO.lean displays both schedules, unchanged FP32 source precision and runtime
scalar epsilon. Only integer row-base multiplication is commuted. -/
namespace VeriTile.Bench.Repairs.L2NormTriton1.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev original := @PairIO.original
abbrev commuted := @PairIO.commuted

/-- Both complete schedules are safe and succeed, every output agrees, and
both preserve the entire exterior frame. Any primitive interpretation and
all epsilon/input payloads are admitted, including all native FP32 words. -/
specification source_equivalence {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) :
    original A rows N B eps ≡[[]] commuted A rows N B eps :=
  ProgramIO.Implementation.of_equivalent (GridIO.candidate A rows N B eps)
    (GridIO.equivalence A rows N B eps)

#stmtSurfaceSubset source_equivalence ⊆
  [FP.Structural.Algebra, original, commuted, Spec.Equiv, ProgramIO.Implementation,
   GridIO.gridIO, GridIO.candidate,
   ProgramIO.Inputs, Spec.RuleEntry, Spec.ProgramSyntax.Statement,
   ProgramIO.Implementation.programSyntax, ProgramIO.InputBindingMode.decoded]
-- No floating transformation law is a parameter of this shared IO judgment.
-- The exact historical whole-state equivalence remains in Misc/SourceEquivalence.lean.
#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.FPEquiv
