import bench.repairs.tritonbench_g.flash_decode2_llama.Proofs.IOPair

namespace VeriTile.Bench.Repairs.FlashDecode2Llama.FPEquiv
open VeriTile Triton FP.Structural FlashDecode2
open scoped VeriTile.Spec
noncomputable section

def original (p : RawNative.Ops) (g : KernelInterface.Geometry) (order : List Nat) :=
  ProgramIO.Implementation.original (IOPair.io p g order) (IO.Lengths g)
def candidate (p : RawNative.Ops) (g : KernelInterface.Geometry) (order : List Nat) :=
  ProgramIO.Implementation.ofVariant (IOPair.variant p g order) (IO.Lengths g)

/-- Share the integer split-row base in the complete unmasked source grid.
All floating words and operation order, runtime lengths, every complete-grid
permutation, actual launch setup and both exterior frames are preserved. -/
specification kernel_equivalent (p : RawNative.Ops) (g : KernelInterface.Geometry)
    (order : List Nat) (coverage : order.Perm (List.range (g.B * g.H))) :
    original p g order ≡[[]] candidate p g order := by
  apply ProgramIO.Implementation.of_equivalentOn
  exact IOPair.equivalent p g order coverage

#axiomsClean kernel_equivalent
end
end VeriTile.Bench.Repairs.FlashDecode2Llama.FPEquiv
