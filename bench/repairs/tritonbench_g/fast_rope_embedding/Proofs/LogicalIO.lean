import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.RealIO

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.LogicalIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The complete previous invocation judgment, with all inputs and outputs. -/
theorem judgment_iff (r : Request) (qStride cosineStride sineStride : Nat)
    (backward : Bool) (row group : Nat) (f : Inputs r → Fin r.heads → Fin (2*r.half) → ℝ) :
    (view r qStride cosineStride sineStride backward row group ⊨ f) ↔
      (Interface.programIO ideal r qStride cosineStride sineStride backward row group).Implements
        (fun raw => Interface.result (f ⟨Interface.query raw,Interface.cosine raw,Interface.sine raw⟩)) := Iff.rfl
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.LogicalIO
