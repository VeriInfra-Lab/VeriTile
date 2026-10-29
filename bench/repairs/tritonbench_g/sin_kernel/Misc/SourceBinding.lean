import bench.repairs.tritonbench_g.sin_kernel.FPEquiv

/-! Optional source-binding checks for the public FP programs.
The original declaration names are retained outside the reader entry. -/
namespace VeriTile.Bench.Repairs.SinKernel.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
noncomputable section

theorem original_source (n : Nat) : (original n).kernel = originalKernel n := rfl
theorem candidate_source (n : Nat) : (indexCommuted n).kernel = candidateKernel n := rfl

#axiomsClean original_source
#axiomsClean candidate_source
end
end VeriTile.Bench.Repairs.SinKernel.FPEquiv
