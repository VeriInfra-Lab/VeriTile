import bench.tritonbench_g.matmul_dequantize_int4.Kernels
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Source
open VeriTile.Triton
noncomputable section
/-! The selected compiler realization replaces the source's add-after-dot site
with a live-carry native dot. This is a compiler graph, not a second source
implementation or an assumed FP algebraic identity. The existing compiler
contracts and instruction proofs still discharge its connection to execution. -/
def carryNode (dtype : FloatDType) (BM BN BK : Nat) : ComputeStmt :=
  .assign .real [BM,BN] "accumulator" (.compute (.alg .fp32
    (.dotAcc dtype dtype .real (batch := [])
      (.ref dtype.toTileDType [BM,BK] "a")
      (.castFloat dtype dtype (.ref dtype.toTileDType [BK,BN] "b"))
      (.ref .real [BM,BN] "accumulator") {})))
def lowerNative (dtype : FloatDType) (BM BN BK : Nat) (kernel : ComputeKernel) : ComputeKernel :=
  .mk kernel.inputs kernel.outputs (kernel.surfaceBody.map fun stmt=>match stmt with
    | .forRangeDyn index start stop stride loop =>
      if index=="k" then .forRangeDyn index start stop stride
        (loop.take 5 ++ [carryNode dtype BM BN BK] ++ loop.drop 6) else stmt
    | _=>stmt)

end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Source
