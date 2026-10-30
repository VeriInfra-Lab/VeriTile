import bench.repairs.tritonbench_g.vector_addition.FPEquiv

/-! Optional source-binding checks for the public FP programs.
The original declaration names are retained outside the reader entry. -/
namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
noncomputable section

abbrev Rules := FP.ScalarArithmetic.Rules
abbrev source (n : Nat) : ComputeKernel := originalKernel n

theorem original_source (n : Nat) : (original n).kernel = source n := rfl

#axiomsClean original_source
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.FPEquiv
