import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Public

namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
open Numerics (Word Tagged leaf)
/-- Whole checked native FP32 word/byte profile. The local multiply/negate/FMA
instruction interpretation is deferred; source success, stores, returned objects,
and the caller byte frame are proved. -/
theorem rope_backward_fp32 (api : API Word Store (BitVec 8)) (p : Numerics.Primitives)
    (r : Request) (m : Source.Memory Tagged) (call : Native.Preparation api leaf r m)
    (codec : ∀ w k,api.encode w k=Numerics.wordByte w k)
    (accepted : Checked.check api leaf r m call=Bool.true) : Checked.FP32Outcome api p r m call :=
  Checked.whole_fp32 api p r m call codec accepted
#axiomsClean rope_backward_fp32
end VeriTile.Bench.Repairs.RopeBackwardTransform
