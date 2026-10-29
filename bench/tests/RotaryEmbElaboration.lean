import bench.tritonbench_g.rotary_emb.Proofs.SourceTypes

/-! Regression for elaborating the complete symbolic 3D source. Returning its
uninstantiated expression from the DSL elaborator previously caused memory
growth at the first K store, after the first 32 statements had elaborated.
This uses the production parser and macro, retaining all four stores and every
symbolic dimension/stride. No diagnostic elaborator is installed here. -/
namespace VeriTile.Bench.Tests.RotaryEmbElaboration
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RotaryEmb.FullSource

set_option maxHeartbeats 200000
set_option maxRecDepth 1000

def source : Source := rotaryFull%[tl.float32, tl.float64, tl.float32]

example (Q K Cos Sin : RegionName)
    (sqb sqh sqd skb skh skd scb scd ssb ssd L HQ HK BH BS D : Nat) :
    (source Q K Cos Sin sqb sqh sqd skb skh skd scb scd ssb ssd
      L HQ HK BH BS D).surfaceBody.length = 34 := rfl

end VeriTile.Bench.Tests.RotaryEmbElaboration
