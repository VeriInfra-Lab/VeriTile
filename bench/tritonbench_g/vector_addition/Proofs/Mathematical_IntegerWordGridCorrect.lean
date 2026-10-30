import bench.tritonbench_g.vector_addition.Proofs.Mathematical_IntegerWordGrid

/-! Public full-grid word result for all 81 integer/bool pairs. Two explicit
input relations bind finite source-width words and cell tags. The original
execution is constructed; no native byte/address/compiler claim is made. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.IntegerWordGridCorrect
open VeriTile.Triton FP.Structural
open IntegerSource (Canonical)
open IntegerWords
open ValueProofs (Parameters)
noncomputable section

specification vector_addition_integer_word_grid_correctness
    {α : Type} [Inhabited α] (M : Algebra α)
    (x y : Canonical) (wx : Nat → Word x) (wy : Nat → Word y)
    (p : Parameters) (initial : State α) (positive : 0 < p.block)
    (hx : InputWords x wx p.x p.n initial.mem) (hy : InputWords y wy p.y p.n initial.mem) :
    ∃ final, IntegerSource.runGrid M x.storage y.storage p initial = some final ∧
      final.mem = wordSpec x y wx wy p initial.mem :=
  grid_correct_words M x y wx wy p initial positive hx hy

#axiomsClean vector_addition_integer_word_grid_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.IntegerWordGridCorrect
