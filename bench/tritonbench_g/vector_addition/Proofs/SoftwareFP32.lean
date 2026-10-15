import bench.tritonbench_g.vector_addition.Proofs.TypedGrid
import VeriTile.Triton.Float.ScalarBackend

/-! Exact software-binary32 value instantiation of the legacy source. The
adapter explicitly supplies its untagged .real addition; scalarFP32 alone
would leave that operation abstract. Native GPU correspondence is separate.
No finiteness restriction, algebraic equality law, or successful-run premise.
-/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.SoftwareFP32
open VeriTile.Triton FP.Structural
open ValueProofs
noncomputable section

local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

def backend (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config) :
    Algebra (FP.Value .fp32) :=
  { rest with binary := fun precision dtype op a b =>
      if precision = none ∧ dtype = .real ∧ op = .add then
        FP.add cfg a b else rest.binary precision dtype op a b }

/-- Independent initial-memory result of binary32 addition, including its
specified rounding and all exceptional bit patterns. -/
def expected (cfg : FP.Config) (p : Parameters) (initial : State (FP.Value .fp32)) :
    RegionName → Nat → Cell (FP.Value .fp32) :=
  fun r o => if r = p.output ∧ o < p.n then
    .mk .real (FP.add cfg ((initial.mem p.x o).read .real)
      ((initial.mem p.y o).read .real)) else initial.mem r o

theorem exact_grid (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (p : Parameters) (initial : State (FP.Value .fp32)) (positive : 0 < p.block) :
    ∃ final, runGrid (backend rest cfg) p initial = some final ∧
      final.mem = expected cfg p initial := by
  obtain ⟨final, he, hm⟩ := grid_correct (backend rest cfg) p initial positive
  refine ⟨final, he, ?_⟩
  simpa [gridExpectedMemory, resultCell, expected, backend] using hm

/-- Concrete binary32 result for the explicitly typed fp32/fp32 profile. -/
def typedExpected (cfg : FP.Config) (p : Parameters) (initial : State (FP.Value .fp32)) :
    RegionName → Nat → Cell (FP.Value .fp32) :=
  fun r o => if r = p.output ∧ o < p.n then
    .mk .fp32 (FP.add cfg ((initial.mem p.x o).read .fp32)
      ((initial.mem p.y o).read .fp32)) else initial.mem r o

theorem exact_typed_grid (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (p : Parameters) (initial : State (FP.Value .fp32)) (positive : 0 < p.block) :
    ∃ final, TypedFloat.runGrid (FP.Software.scalarFP32 rest cfg) .fp32 .fp32 .fp32 p initial =
      some final ∧ final.mem = typedExpected cfg p initial := by
  obtain ⟨final, he, hm⟩ := TypedFloat.grid_correct (FP.Software.scalarFP32 rest cfg)
    .fp32 .fp32 .fp32 p initial positive
  refine ⟨final, he, ?_⟩
  simpa [TypedFloat.gridExpectedMemory, TypedFloat.resultCell, TypedFloat.castValue,
    typedExpected, FP.Software.scalarFP32, FP.Software.fp32Context,
    ofFloat, toFloat] using hm

#specNonCircular expected avoiding [source, runGrid, FP.Structural.exec]
#axiomsClean exact_grid
#specNonCircular typedExpected avoiding [TypedFloat.source, TypedFloat.runGrid, FP.Structural.exec]
#axiomsClean exact_typed_grid
end
end VeriTile.Bench.TritonBenchG.VectorAddition.SoftwareFP32
