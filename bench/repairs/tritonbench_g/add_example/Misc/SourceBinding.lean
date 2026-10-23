import bench.repairs.tritonbench_g.add_example.FPEquiv

/-! Optional source-binding checks for the public FP programs.
The original declaration names are retained outside the reader entry. -/
namespace VeriTile.Bench.Repairs.AddExample.Contiguous.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
noncomputable section

theorem original_source (n : Nat) : (original n).kernel = source n := rfl

#axiomsClean original_source
end
end VeriTile.Bench.Repairs.AddExample.Contiguous.FPEquiv
