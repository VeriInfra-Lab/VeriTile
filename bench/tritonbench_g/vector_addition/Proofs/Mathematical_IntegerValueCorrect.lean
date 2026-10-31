import bench.tritonbench_g.vector_addition.Kernels
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

#specNonCircular spec avoiding [VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel, FP.Structural.exec, FP.Structural.run]
end
end VeriTile.Bench.TritonBenchG.VectorAddition.IntegerValueCorrect
