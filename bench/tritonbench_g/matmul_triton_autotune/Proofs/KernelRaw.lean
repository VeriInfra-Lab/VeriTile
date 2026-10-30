import bench.tritonbench_g.matmul_triton_autotune.Proofs.Refinement

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
open VeriTile.Triton FP.Structural Whole

/-- Raw source geometry and selected arithmetic configuration. -/
structure Geometry where
  (A B C : RegionName)
  (rows inner columns : Nat)
  (strideAM strideAK strideBK strideBN : Nat)
  activation : String
  config : Config

/-- The source interpreter only uses these geometry fields. The legacy host
admission fields are fixed internally and are absent from kernel inputs. -/
def Geometry.request (g : Geometry) : Whole.Request where
  A := g.A; B := g.B; C := g.C
  rows := g.rows; inner := g.inner; innerB := g.inner; columns := g.columns
  strideAM := g.strideAM; strideAK := g.strideAK
  strideBK := g.strideBK; strideBN := g.strideBN
  aContiguous := Bool.true
  activation := g.activation
  target := ""
  config := g.config

end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
