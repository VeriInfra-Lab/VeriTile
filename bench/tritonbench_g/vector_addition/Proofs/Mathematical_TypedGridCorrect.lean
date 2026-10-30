import bench.tritonbench_g.vector_addition.Proofs.Mathematical_TypedValueCorrect

/-! Registered public surface for the independently reviewed typed floating
grid theorem. The five storage formats give 25 input pairs. Addition and
distinct-format casts remain arbitrary backend primitives. This wrapper adds
no proof premise and does not establish native dtype, memory or host coverage.
-/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedGridCorrect
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters)
noncomputable section

specification vector_addition_typed_grid_correctness
    {α : Type} [Inhabited α] (M : Algebra α)
    (x y : TypedFloat.Storage) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) :
    ∃ final,
      TypedFloat.runGrid M x.dtype y.dtype (TypedFloat.promote x y).dtype p initial =
        some final ∧ final.mem = TypedValueCorrect.spec M x y p initial := by
  exact TypedValueCorrect.correct M x y p initial positive

#axiomsClean vector_addition_typed_grid_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedGridCorrect
