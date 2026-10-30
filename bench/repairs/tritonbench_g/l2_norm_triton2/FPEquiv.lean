import bench.repairs.tritonbench_g.l2_norm_triton2.IO
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Pair

/-! Public full-grid equivalence of the selected FP32 sources. The existing
candidate changes only integer row-base multiplication. Primitive floating
operations, exceptional payloads and the actual runtime epsilon are unchanged. -/
namespace VeriTile.Bench.Repairs.L2NormTriton2.FPEquiv
open VeriTile.Triton FP.Structural FloatingIO
open scoped VeriTile.Spec
noncomputable section

/-- Both complete forward schedules are safe, succeed, agree on every output
and preserve the entire exterior frame for every primitive interpretation. -/
specification forward_equivalence {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) :
    forwardOriginal A rows N B eps ≡[[]] forwardCandidate A rows N B eps :=
  Pair.forward A rows N B eps

/-- Both complete backward schedules have identical observed cells, with
source access safety and the full frame. All input and epsilon words remain admitted. -/
specification backward_equivalence {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) :
    backwardOriginal A rows N B eps ≡[[]] backwardCandidate A rows N B eps :=
  Pair.backward A rows N B eps

#stmtSurfaceSubset forward_equivalence ⊆
  [FP.Structural.Algebra,forwardOriginal,forwardCandidate,Spec.Equiv,
   ProgramIO.Implementation,ProgramIO.Implementation.programSyntax,
   L2NormTriton1.GridIO.gridIO,GridIO.forwardIO,ProgramIO.Inputs,
   ProgramIO.InputBindingMode.decoded,Spec.RuleEntry,Spec.ProgramSyntax.Statement]
#stmtSurfaceSubset backward_equivalence ⊆
  [FP.Structural.Algebra,backwardOriginal,backwardCandidate,Spec.Equiv,
   ProgramIO.Implementation,ProgramIO.Implementation.programSyntax,GridIO.gridIO,
   ProgramIO.Inputs,ProgramIO.InputBindingMode.decoded,Spec.RuleEntry,
   Spec.ProgramSyntax.Statement]
-- This shared IO judgment has no floating transformation assumptions.
-- The historical full-state equivalences and their empty assumption audits
-- remain ordinary results in Misc/SourceEquivalence.lean.
#axiomsClean forward_equivalence
#axiomsClean backward_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.FPEquiv
