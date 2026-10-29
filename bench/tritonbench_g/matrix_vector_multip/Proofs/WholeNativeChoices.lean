import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeNative

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeNativeChoices
open WholeNative
noncomputable section

/-- Every possible shared/global last writer is retained as a structural
choice. A selector contains only discrete bits, never computed FP values. -/
inductive Choices (columns : Nat) where
  | leaf (column : Fin columns)
  | add (left right : Choices columns)
  | choose (site : Nat) (left right : Choices columns)

def Choices.leaves : Choices columns → List (Fin columns)
  | .leaf j => [j]
  | .add left right => left.leaves++right.leaves
  | .choose _ left _ => left.leaves

def Choices.resolve (winner : Nat → Bool) : Choices columns → AddTree columns
  | .leaf j => .leaf j
  | .add left right => .add (left.resolve winner) (right.resolve winner)
  | .choose site left right => if winner site then right.resolve winner else left.resolve winner

/-- Equal leaf multisets are purely source-coordinate coverage facts. They
make no assertion that different ordered floating trees have equal values. -/
def Choices.Covered : Choices columns → Prop
  | .leaf _ => True
  | .add left right => left.Covered ∧ right.Covered
  | .choose _ left right => left.Covered ∧ right.Covered ∧ right.leaves.Perm left.leaves

theorem Choices.resolve_coverage (t : Choices columns) (covered : t.Covered) (winner : Nat → Bool) :
    (t.resolve winner).leaves.Perm t.leaves := by
  induction t with
  | leaf j => exact List.Perm.refl _
  | add left right ihLeft ihRight =>
    exact (ihLeft covered.1).append (ihRight covered.2)
  | choose site left right ihLeft ihRight =>
    unfold resolve
    split
    · exact (ihRight covered.2.1).trans covered.2.2
    · exact ihLeft covered.1

structure Plan (rows columns : Nat) where
  tree : Fin rows → Choices columns
  covered : ∀ row,(tree row).Covered
  complete : ∀ row,(tree row).leaves.Perm (List.finRange columns)

def Plan.graph (p : Plan rows columns) (winner : Nat → Bool)
    (orientation : Fin rows → Fin columns → Bool) : Graph rows columns where
  tree:=fun row => (p.tree row).resolve winner
  reverseFMA:=orientation
  complete:=fun row => ((p.tree row).resolve_coverage (p.covered row) winner).trans (p.complete row)

/-- Every possible last-writer choice gives a full graph and is therefore
covered by the existing source/native lowering and complete host theorem. -/
theorem Plan.every_selection_complete (p : Plan rows columns) (winner : Nat → Bool)
    (orientation : Fin rows → Fin columns → Bool) (row : Fin rows) :
    ((p.graph winner orientation).tree row).leaves.Nodup ∧
      ∀ col : Fin columns,col∈((p.graph winner orientation).tree row).leaves :=
  (p.graph winner orientation).exactly_once row

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeNativeChoices
