import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.IOPair

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPEquiv
open VeriTile Triton FP.Structural NativeGraph
open scoped VeriTile.Spec
noncomputable section

def original (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) :=
  ProgramIO.Implementation.original (NativeIO.program R h backward epsilon) (Reference.nativeDomain h backward)

def candidate (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) :=
  ProgramIO.Implementation.ofVariant (IOPair.variant R h backward epsilon) (Reference.nativeDomain h backward)

/-- The complete original and preserved row-index candidate are safe, succeed,
produce identical active typed output cells and preserve both exterior frames.
Runtime epsilon, regular/Gemma modes and the weight-before-input backward grids
are retained, including the empty-row zero-DW domain and arbitrary FP32 words.
No floating computation is rewritten. -/
specification kernel_equivalent (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) :
    original R h backward epsilon ≡[[]] candidate R h backward epsilon := by
  apply ProgramIO.Implementation.of_equivalentOn
  exact IOPair.equivalent R h backward epsilon

#print_fp_assumptions kernel_equivalent
#axiomsClean kernel_equivalent
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPEquiv
