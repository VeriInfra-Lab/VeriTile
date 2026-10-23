import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.RealExecution
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealIO

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.RealFlat
open VeriTile.Triton RealExecution
namespace Original
abbrev sourceIO := VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealIO.sourceIO
abbrev expected := VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealIO.expected
abbrev Implements := @VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealIO.IO.Implements
end Original

/-- Reuse the original complete per-program flat-memory execution, including
signed sentinel, every loop iteration, output readback and all other cells.
The repaired wrapper's fresh snapshot allocation supplies this disjointness. -/
theorem complete (h : Call) :
    Original.Implements (Original.sourceIO (parameters h) h.storage h.outputType Bool.false)
      (Original.expected (parameters h)) := by
  exact VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealIO.implements (parameters h) h.storage h.outputType Bool.false
    (by change 0<256*64; decide) (by change 0<1; decide) h.rowsCovered

#axiomsClean complete
end VeriTile.Bench.Repairs.BgmvShrinkKernel.RealFlat
