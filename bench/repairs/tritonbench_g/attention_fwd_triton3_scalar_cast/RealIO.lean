import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.IO

/-! Finite six-port IO for the complete ordinary grid. The channel coordinates
retain the program, row and feature dimensions; repeated read coordinates are
bound simultaneously. The mathematical source explicitly erases the original
FP32 quotation. Carry ports alias their own outputs, and L is preserved at END.
The separate existing flat-memory and Tensor transport remains in Auxiliary.
-/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section

/-- Independent maximum and weighted sums over the signed-window key set,
with the exact inherited INIT, no-loop, padded-row and END conventions. -/
def formula (c : Call) (v : Inputs c) : Outputs c := fun o j =>
  let point := (outputCoordinates c o).symm j
  (RealContract.expected c.BM c.BN c.D c.finish (c.tile point.1)
    c.params (localInputs c v point.1)).channel o point.2

#specNonCircular formula avoiding
  [Kernels.source,exec,execR,Kernel.GridLaunchedOrdinary,AlgorithmExecution]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
