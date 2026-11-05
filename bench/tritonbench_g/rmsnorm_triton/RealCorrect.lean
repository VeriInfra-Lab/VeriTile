import bench.tritonbench_g.rmsnorm_triton.IO
import bench.tritonbench_g.rmsnorm_triton.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.RmsnormTriton.Correct
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
open scoped VeriTile.Spec VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

/-- The complete 2D grid implements RMS normalization on every logical row.
IO exposes all input/weight/output strides and has no private scratch. The
shared judgment includes actual masked-access safety, successful execution,
all row outputs and the raw-cell exterior frame. Active output uniqueness
retains singleton rows with zero column stride. Physical separation and
native/Tensor applications remain explicit in the ordinary proof adapters. -/
specification kernel_correct (g : KernelGrid.Geometry)
    (_separation : Separation g.parameters) (unique : OutputUnique g.parameters)
    (work : Schedule g.parameters) :
    Spec.Real (RmsIO.logicalIO g work ⊨ fun values cell =>
      let xs := values.1
      let weights := values.2
      let batch := cell.1.1
      let row := cell.1.2
      let k := cell.2
      (xs batch row k * (Real.sqrt
        ((∑ j : Fin g.width, xs batch row j*xs batch row j)/(g.width:ℝ)+g.eps))⁻¹) * weights k) :=
  (IOCorrect.logical_iff g work).mpr (IOCorrect.complete g unique work)

#guard_msgs (drop info) in
#specNonCircular RmsIO.rms avoiding [kernel,exec,execR,runAlgorithmSchedule]
#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormTriton.Correct
