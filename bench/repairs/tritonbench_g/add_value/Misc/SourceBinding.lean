import bench.repairs.tritonbench_g.add_value.FPEquiv
import bench.repairs.tritonbench_g.add_value.Proofs.Source

/-! Optional source-binding checks for the public FP programs.
The original declaration names are retained outside the reader entry. -/
namespace VeriTile.Bench.Repairs.AddValue.Contiguous.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
noncomputable section

/-- The public structural source and rewrite source are literally the same
ComputeKernel; no erased/profiled substitute is used for FP equivalence. -/
theorem original_source (n : Nat) : (original n).kernel = source n := rfl

/-- The displayed FP32 kernel lowers to its finite instruction sequence.
This is source binding, not an IEEE arithmetic premise. -/
theorem source_lowering (n : Nat) :
    Source.lowerComputeBody (source n).surfaceBody =
      some (Source.sourceCode "x" "out" n) :=
  Source.original_ast_lowering "x" "out" n

#axiomsClean original_source
#axiomsClean source_lowering
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.FPEquiv
