import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.LegacyPastedSurface
import bench.tritonbench_g.rms_norm_triton.Proofs.LegacyKernels

/-! Source connections to the retained historical typed family. The current
selected source definitions are authoritative in the public Kernels modules. -/
namespace VeriTile.Bench.Repairs.RmsNormTriton.LegacyPastedSurface
open VeriTile.Triton

theorem legacy_original_identity (Y X W : RegionName) (N B : Nat) (eps : ℝ) :
    kernel Y X W N B eps =
      VeriTile.Bench.TritonBenchG.RmsNormTriton.typed .fp32 .fp32 Bool.false
        Y X W N 1 N 1 N eps B := rfl


/-- Equality of the complete original source function, retaining all parameters. -/
theorem original_function_identity :
    @kernel = fun Y X W N B eps =>
      VeriTile.Bench.TritonBenchG.RmsNormTriton.typed .fp32 .fp32 Bool.false
        Y X W N 1 N 1 N eps B := rfl

/-- Equality of the complete candidate function, including every typed statement. -/
theorem indexCommuted_function_identity :
    @indexCommutedKernel = fun Y X W N B eps =>
      VeriTile.Bench.TritonBenchG.RmsNormTriton.typed .fp32 .fp32 Bool.true
        Y X W N 1 N 1 N eps B := rfl

#axiomsClean legacy_original_identity
#axiomsClean original_function_identity
#axiomsClean indexCommuted_function_identity
end VeriTile.Bench.Repairs.RmsNormTriton.LegacyPastedSurface
