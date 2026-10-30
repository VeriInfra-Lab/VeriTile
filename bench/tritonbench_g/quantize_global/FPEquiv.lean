import bench.tritonbench_g.quantize_global.Proofs.Pair

/-! The same FP32 div.rn/multiply/clamp/llrint64/INT8 scalar graph under any
nonempty sequences of the retained autotune configurations. Both actual grids
succeed safely, agree on every typed output and preserve the exterior memory.
Primitive callbacks are unchanged on both sides; numerical transformation
assumptions: none. This is a fixed integer-stage observable IO equivalence. -/
namespace QuantizeGlobalRepair.FPEquivalence
open VeriTile VeriTile.Triton FP.Structural QuantizeIO
open scoped VeriTile.Spec
noncomputable section

specification configurations_equivalent {α : Type} [Inhabited α]
    (rest : Algebra α) (a : Scalar α) (p : Parameters) (sep : p.Separated)
    (left right : Nat → List Config)
    (leftNonempty : ∀ k, k < segmentCount p.n → left k ≠ [])
    (rightNonempty : ∀ k, k < segmentCount p.n → right k ≠ []) :
    original rest a p left ≡[[]] candidate rest a p left right :=
  PairProof.equivalent rest a p sep left right leftNonempty rightNonempty

#print_fp_assumptions configurations_equivalent
#axiomsClean configurations_equivalent
#stmtSurfaceSubset configurations_equivalent ⊆
  [Spec.Equiv, original, candidate, FP.Structural.Algebra, Scalar, Parameters,
    Parameters.Separated, Parameters.n, Config, QuantizeGlobalFinite.Config, segmentCount,
    IntegerProgramIO.Implementation, IntegerProgramIO.Implementation.instProgramSyntaxOfInhabited,
    io,IntegerProgramIO.Inputs,Spec.RuleEntry,Spec.ProgramSyntax.Statement]
end
end QuantizeGlobalRepair.FPEquivalence
