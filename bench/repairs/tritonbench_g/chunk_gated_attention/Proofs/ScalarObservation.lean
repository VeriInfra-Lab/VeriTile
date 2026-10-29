import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.ScalarNative

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section

/-- The original pointwise scalar realization laws, with raw-bit identity. -/
abbrev ScalarLaws (actual reference : ScalarNative.Primitives) :=
  ScalarGraph.ScalarTransport id actual.scalar reference.scalar

theorem scalar_eval_eq (actual reference : ScalarNative.Primitives)
    (laws : ScalarLaws actual reference) (e : NativeGraph.Expr) :
    ScalarNative.eval actual e = ScalarNative.eval reference e :=
  ScalarNative.transport id actual.scalar reference.scalar laws e

#axiomsClean scalar_eval_eq
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
