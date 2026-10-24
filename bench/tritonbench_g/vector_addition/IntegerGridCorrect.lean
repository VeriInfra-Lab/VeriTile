import bench.tritonbench_g.vector_addition.IntegerValueCorrect

/-! Public registration of the frozen 81-pair integer/bool grid theorem.
This delegates without changing its premises or concrete BitVec result.
Logical cell decoding is separate from native byte/storage correspondence. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.IntegerGridCorrect
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters)
open IntegerSource (Canonical)
noncomputable section

specification vector_addition_integer_grid_correctness
    {α : Type} [Inhabited α] (M : Algebra α)
    (x y : Canonical) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) :
    ∃ final, IntegerSource.runGrid M x.storage y.storage p initial = some final ∧
      final.mem = IntegerValueCorrect.spec x y p initial := by
  exact IntegerValueCorrect.correct M x y p initial positive

#axiomsClean vector_addition_integer_grid_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.IntegerGridCorrect
