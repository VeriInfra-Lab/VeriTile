import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.IO
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.FPEquiv
open VeriTile.Triton FP.Structural HarmonicIO KernelScalar
noncomputable section

/-- Complete source-to-native observation under the existing concrete FP32
separate-rounding model. This is not a floating transformation assumption. -/
theorem forward (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    (forwardIO (forwardBackend selectedProfile) p).Implements
      (forwardTarget (forwardBackend selectedProfile) p nativeForward) :=
  IOCorrect.native_forward p fit

/-- All upstream channels and the actual initial gradient are retained; the
source reads and adds the old gradient before storing the three results. -/
theorem backward (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    (backwardIO (backwardBackend selectedProfile) p).Implements
      (backwardTarget (backwardBackend selectedProfile) p nativeBackward) :=
  IOCorrect.native_backward p fit

#specNonCircular nativeForward avoiding [forwardKernel, backwardKernel, FP.Structural.exec]
#specNonCircular nativeBackward avoiding [forwardKernel, backwardKernel, FP.Structural.exec]
#axiomsClean forward
#axiomsClean backward
#stmtSurfaceSubset forward ⊆
  [ProgramIO.Implements, forwardIO, forwardTarget, nativeForward, forwardBackend,
   selectedProfile, Packing.Word, Parameters, Parameters.rows, Parameters.columnOffset, Parameters.outputWidth, TileIndex]
#stmtSurfaceSubset backward ⊆
  [ProgramIO.Implements, backwardIO, backwardTarget, nativeBackward, backwardBackend,
   selectedProfile, Packing.Word, Parameters, Parameters.rows, Parameters.columnOffset, Parameters.outputWidth, TileIndex]
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.FPEquiv
