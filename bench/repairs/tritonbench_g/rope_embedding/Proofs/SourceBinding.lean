import bench.repairs.tritonbench_g.rope_embedding.Proofs.Source
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton

/-- The whole actual source, including its true ports, decomposes into the
proved prelude and complete dynamic loop. The old phantom group-size assignment
is absent: the selected Python declares it as a constexpr argument. The Real,
FP, safety and frame proofs execute this new actual body directly. -/
theorem selected_source_binding (p : Source.Parameters) (back : Bool) :
    Source.source p back = ComputeKernel.mk [p.input,p.cosine,p.sine] [p.input]
      (Source.prelude p back ++
        [.forRangeDyn "k" (.ref .nat [] "head_start") (.ref .nat [] "head_end")
          (.constNat 1) (Source.body p back)]) := by
  cases back <;> rfl

#axiomsClean selected_source_binding
end VeriTile.Bench.Repairs.RopeEmbedding
