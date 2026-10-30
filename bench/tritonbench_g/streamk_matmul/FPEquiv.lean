import bench.tritonbench_g.streamk_matmul.Proofs.Pair
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.StreamK.TileChecked.FPEquiv
open Whole001 Source VeriTile Triton
open scoped VeriTile.Spec
noncomputable section

def original (reference : Native.Reference) (p : Parameters) :
    ProgramView.Implementation reference p := ⟨kernels⟩

def tileArea (reference : Native.Reference) (p : Parameters) :
    ProgramView.Implementation reference p := ⟨areaKernels⟩

/-- Actual complete first/reduce/full programs and all six helper sources.
Fold constexpr tile area in scratch addresses; all FP32 MMA/atomic ordering,
FP16 full-tile round trips, source reads, masks and frames are retained. -/
specification tile_area_equiv (reference : Native.Reference) (p : Parameters) :
    original reference p ≡[([] : Spec.Assumptions _)] tileArea reference p :=
  Spec.FloatingPoint.ofStructural rfl rfl (AreaEquivalence.equivalent reference p)

#axiomsClean tile_area_equiv
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.FPEquiv
