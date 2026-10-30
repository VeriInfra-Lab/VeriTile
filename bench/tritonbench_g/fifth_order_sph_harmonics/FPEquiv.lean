import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.IOPair

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

def original (d : Grid.Direction) (p : Parameters) :=
  ProgramIO.Implementation.original (IOPair.io d p) (fun _ => True)
def candidate (d : Grid.Direction) (p : Parameters) :=
  ProgramIO.Implementation.ofVariant (IOPair.variant d p) (fun _ => True)

/-- Reuse of identically rounded left-associated powers in both complete
source grids. All FP32 words, eleven upstream channels, arbitrary old gradient
words, the actual grid and both exterior memory frames are retained. -/
specification kernel_equivalent (d : Grid.Direction) (p : Parameters)
    (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    original d p ≡[[]] candidate d p := by
  apply ProgramIO.Implementation.of_equivalent
  exact IOPair.equivalent d p fit

#axiomsClean kernel_equivalent
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.FPEquiv
