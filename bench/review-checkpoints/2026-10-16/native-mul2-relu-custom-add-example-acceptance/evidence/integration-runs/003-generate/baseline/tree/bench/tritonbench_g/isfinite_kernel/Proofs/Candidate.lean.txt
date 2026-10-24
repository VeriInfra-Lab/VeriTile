import bench.tritonbench_g.isfinite_kernel.Proofs.Source

/-! Exact structural splits of the independent complete candidate source. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
open VeriTile.Triton

def candidateLoop32 (p : Parameters) : List ComputeStmt := loopBody (candidate32 p)
def candidateLoop64 (p : Parameters) : List ComputeStmt := loopBody (candidate64 p)

def tileIdStmt (swapped : Bool) : ComputeStmt :=
  .assign .nat [] "tile_id" (.alg (.add .nat .nil (.ref .nat [] "pid")
    (if swapped then .mul .nat .nil (.ref .nat [] "num_ctas") (.ref .nat [] "j")
      else .mul .nat .nil (.ref .nat [] "j") (.ref .nat [] "num_ctas"))))

theorem candidate32_split (p : Parameters) :
    (candidate32 p).surfaceBody = before p ++
      [.ifThenElse (.constBool p.oneTile) (oneBody32 p)
        [numCtas, .forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
          (.constNat 1) (candidateLoop32 p)]] := rfl

theorem candidate64_split (p : Parameters) :
    (candidate64 p).surfaceBody = before p ++
      [.ifThenElse (.constBool p.oneTile) (oneBody64 p)
        [numCtas, .forRangeDyn "j" (.constNat 0) (.constNat p.iterations)
          (.constNat 1) (candidateLoop64 p)]] := rfl

theorem loop32_split (p : Parameters) :
    loopBody32 p = tileIdStmt Bool.false :: (loopBody32 p).drop 1 := rfl

theorem loop64_split (p : Parameters) :
    loopBody64 p = tileIdStmt Bool.false :: (loopBody64 p).drop 1 := rfl

theorem candidateLoop32_split (p : Parameters) :
    candidateLoop32 p = tileIdStmt Bool.true :: (loopBody32 p).drop 1 := rfl

theorem candidateLoop64_split (p : Parameters) :
    candidateLoop64 p = tileIdStmt Bool.true :: (loopBody64 p).drop 1 := rfl

#axiomsClean candidate32_split
#axiomsClean candidate64_split
#axiomsClean loop32_split
#axiomsClean loop64_split
#axiomsClean candidateLoop32_split
#axiomsClean candidateLoop64_split
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.Parametric
