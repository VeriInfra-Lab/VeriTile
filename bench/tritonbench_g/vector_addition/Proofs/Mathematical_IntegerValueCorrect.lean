import bench.tritonbench_g.vector_addition.Proofs.Mathematical_IntegerGrid
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_IntegerProfiles

/-! Public contract for all 81 canonical integer/bool pairs. The specification
is a concrete initial-memory bitvector formula. No floating callback appears
in it. The interpreter algebra parameter is unused by this integer graph.
Native memory representations, signed address overflow and sanitizer errors,
compiler admission and actual GPU scheduling still require separate bridges.
This theorem alone does not establish whole-original-program acceptance. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.IntegerValueCorrect
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters)
open IntegerSource (Canonical)
noncomputable section

def spec {α : Type} [Inhabited α] (x y : Canonical)
    (p : Parameters) (initial : State α) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < p.n then
    IntegerSource.resultCell x.storage y.storage p initial.mem o
    else initial.mem r o

/-- Complete ordered launch including tail mask and the frame at every other
address. Equal logical x/y/output regions are permitted. -/
theorem correct {α : Type} [Inhabited α] (M : Algebra α)
    (x y : Canonical) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) :
    ∃ final, IntegerSource.runGrid M x.storage y.storage p initial = some final ∧
      final.mem = spec x y p initial := by
  exact IntegerSource.grid_correct M x.storage y.storage p initial positive

#specNonCircular spec avoiding [IntegerSource.source, IntegerSource.runGrid,
  IntegerSource.runGridPrefix, FP.Structural.exec]
#axiomsClean correct
end
end VeriTile.Bench.TritonBenchG.VectorAddition.IntegerValueCorrect
