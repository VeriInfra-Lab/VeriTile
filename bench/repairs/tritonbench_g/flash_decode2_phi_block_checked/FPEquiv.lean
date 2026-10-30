import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.IOPair

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.FPEquiv
open VeriTile Triton FP.Structural FlashDecode2
open scoped VeriTile.Spec
noncomputable section

def original (g : PhiIO.Geometry) (order : List Nat) :=
  ProgramIO.Implementation.original (IOPair.io g order) (PhiIO.Fits g)
def candidate (g : PhiIO.Geometry) (order : List Nat) :=
  ProgramIO.Implementation.ofVariant (IOPair.variant g order) (PhiIO.Fits g)

/-- Share the integer split-row base in the complete masked source grid.
All floating words and operation order, runtime lengths and padded-lane masks, every complete-grid
permutation, actual launch setup and both exterior frames are preserved. -/
specification kernel_equivalent (g : PhiIO.Geometry) (width : g.D ≤ g.BD)
    (order : List Nat) (coverage : order.Perm (List.range (g.B * g.H))) :
    original g order ≡[[]] candidate g order := by
  apply ProgramIO.Implementation.of_equivalentOn
  exact IOPair.equivalent g width order coverage

#axiomsClean kernel_equivalent
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.FPEquiv
