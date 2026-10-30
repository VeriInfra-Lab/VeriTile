import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.RmsnormTriton.IOCorrect
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
open KernelGrid RmsIO
noncomputable section

/-- Compose whole-source safety with the retained two-pass row theorem over
exactly the selected full 2D schedule. No execution premise is supplied. -/
theorem complete (g : Geometry) (unique : OutputUnique g.parameters)
    (work : Schedule g.parameters) :
    (io g).RegionalScheduledImplements (schedule g work) (target g) := by
  apply UKernelIO.RegionalScheduledImplements.intro
  · intro s vals _ bounds bounded
    exact schedule_safe g unique work bounds s vals bounded
  · exact complete_run g unique work

#axiomsClean complete
end
end VeriTile.Bench.Repairs.RmsnormTriton.IOCorrect
