import bench.repairs.tritonbench_g.f8_conversion_utils.Kernels
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.F8ConversionUtils.KernelReal
open VeriTile.Torch Checked
noncomputable section

/-- Complete source execution for the independent mathematical cast-erasure
formula. No finite numerical domain or lossless FP8 round trip is asserted. -/
theorem complete_kernel_real_formula (p : Source.Profile)
    (Y X : VeriTile.Triton.RegionName) (N B : Nat) (initial : VeriTile.Triton.BlockState) :
    ∃ final, VeriTile.Triton.exec (Source.kernel p Y X N B).eraseDType.toAlgKernel initial=some final ∧
      (∀ j : Fin B,initial.pid*B+j.val<N →
        final.readMem Y (initial.pid*B+j.val)=conversionReal (initial.readMem X (initial.pid*B+j.val))) ∧
      (∀ r i,(r≠Y ∨ ∀ j : Fin B,initial.pid*B+j.val<N → i≠initial.pid*B+j.val) →
        final.mem r i=initial.mem r i) := by
  simpa only [Checked.real_source,conversionReal] using
    RealSource.executes p.direction p.encoder p.format Y X N B initial

#axiomsClean complete_kernel_real_formula

end
end VeriTile.Bench.Repairs.F8ConversionUtils.KernelReal
