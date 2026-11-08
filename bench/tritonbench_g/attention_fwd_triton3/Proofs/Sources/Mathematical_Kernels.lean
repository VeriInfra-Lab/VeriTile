import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Selected

/-! Compatibility name for the only selected repaired source quotation.
The old FP16 and reciprocal quotations are retired; source proofs use
-/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.FP32Source
noncomputable abbrev original := VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Quoted.kernel
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.FP32Source
