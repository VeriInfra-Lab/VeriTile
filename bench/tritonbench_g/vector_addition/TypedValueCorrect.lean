import bench.tritonbench_g.vector_addition.Proofs.TypedGrid
import bench.tritonbench_g.vector_addition.Proofs.DTypeScope

/-! Public bounded-scope contract: the original per-program graph, with
explicit promotion and store casts for the five represented floating storage
formats, composed over the entire ordered launch grid. Floating primitives
are arbitrary; no Real algebra or rewrite equality is assumed. Native dtype
support, allocation/layout, finite-width launch metadata and concurrent GPU
execution remain outside this structural contract. This does not establish
whole-original-program acceptance. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedValueCorrect
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters)
noncomputable section

/-- Independent initial-memory formula for every address, including frame. -/
def spec {α : Type} [Inhabited α] (M : Algebra α)
    (x y : TypedFloat.Storage) (p : Parameters) (initial : State α) :
    RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < p.n then
    TypedFloat.resultCell M x.dtype y.dtype (TypedFloat.promote x y).dtype p initial.mem o
    else initial.mem r o

theorem correct {α : Type} [Inhabited α] (M : Algebra α)
    (x y : TypedFloat.Storage) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) :
    ∃ final, TypedFloat.runGrid M x.dtype y.dtype (TypedFloat.promote x y).dtype p initial =
      some final ∧ final.mem = spec M x y p initial := by
  exact TypedFloat.grid_correct M x.dtype y.dtype (TypedFloat.promote x y).dtype p initial positive

#specNonCircular spec avoiding [TypedFloat.source, TypedFloat.materialized,
  TypedFloat.runGrid, FP.Structural.exec]
#axiomsClean correct
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedValueCorrect
